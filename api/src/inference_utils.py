from typing import Dict
import numpy as np 

NULL_CHAR = '\x00'

# Map Arabic character variants that are absent from the training alphabet to their
# nearest known equivalent. This prevents unknown chars from silently mapping to EOS.
_ARABIC_NORMALISATION: dict[str, str] = {
    'أ': 'ا',  # أ → ا  (alef with hamza above → plain alef)
    'إ': 'ا',  # إ → ا  (alef with hamza below → plain alef)
    'آ': 'ا',  # آ → ا  (alef with madda → plain alef)
    'ٱ': 'ا',  # ٱ → ا  (alef wasla → plain alef)
    'ى': 'ي',  # ى → ي  (alef maqsura → ya)
    'ة': 'ه',  # ة → ه  (ta marbuta → ha)
}


def normalise_arabic(text: str) -> str:
    """Replace unsupported Arabic character variants with their nearest alphabet equivalent."""
    return ''.join(_ARABIC_NORMALISATION.get(c, c) for c in text)


def construct_alphabet_list(alphabet_string: str) -> list[str]:
    if not isinstance(alphabet_string, str):
        raise TypeError("alphabet_string must be a string") 
    
    char_list = list(alphabet_string) 
    return [NULL_CHAR] + char_list 

def get_alphabet_map(alphabet_list: list[str]) -> Dict[str, int]:
    """creates a char to index map from full alphabet list"""
    return {char: idx for idx, char in enumerate(alphabet_list)}  

def encode_text(text: str, char_to_index_map: Dict[str, int], 
                max_length: int, add_eos: bool = True, eos_char_index: int = 0
                ) -> tuple[np.ndarray, int]:
    """Encode a text string into a sequence of integer indices"""
    encoded = [char_to_index_map.get(c, eos_char_index) for c in text] 
    if add_eos:
        encoded.append(eos_char_index) 

    true_length = len(encoded)

    if true_length <= max_length: 
        padded_encoded = np.full(max_length, eos_char_index, dtype=np.int64) 
        padded_encoded[:true_length] = encoded 
    else:
        padded_encoded = np.array(encoded[:max_length], dtype=np.int64) 
        true_length = max_length 
    
    return np.array([padded_encoded]), true_length


def trim_stroke_noise(strokes: list[list[float]], min_segment_pts: int = 3, max_consecutive_tiny: int = 2) -> list[list[float]]:
    """Remove noisy trailing micro-segments that appear after the letter is complete.

    Identifies pen-down segments shorter than min_segment_pts and trims output at
    the first run of max_consecutive_tiny such segments in a row.
    """
    if not strokes:
        return strokes

    # Build list of pen-down segment sizes and their start indices
    segments: list[tuple[int, int]] = []  # (start_idx, size)
    in_seg = False
    seg_start = 0
    for i, s in enumerate(strokes):
        pen_down = s[2] < 0.5
        if pen_down and not in_seg:
            seg_start = i
            in_seg = True
        elif not pen_down and in_seg:
            segments.append((seg_start, i - seg_start))
            in_seg = False
    if in_seg:
        segments.append((seg_start, len(strokes) - seg_start))

    if not segments:
        return strokes

    # Find first run of max_consecutive_tiny tiny segments
    tiny_run = 0
    cutoff_start = -1
    for seg_idx, (start, size) in enumerate(segments):
        if size < min_segment_pts:
            tiny_run += 1
            if tiny_run >= max_consecutive_tiny:
                # cut before the first tiny segment in this run
                first_tiny = seg_idx - (max_consecutive_tiny - 1)
                cutoff_start = segments[first_tiny][0]
                break
        else:
            tiny_run = 0

    if cutoff_start == -1:
        return strokes
    return strokes[:cutoff_start]


def is_degenerate_strokes(strokes: list[list[float]], max_aspect: float = 20.0, min_range: float = 1.5) -> bool:
    """Return True if strokes form a near-straight degenerate line (bias collapse artifact)."""
    if len(strokes) < 5:
        return True
    xs = [s[0] for s in strokes]
    ys = [s[1] for s in strokes]
    x_range = max(xs) - min(xs)
    y_range = max(ys) - min(ys)
    if x_range < min_range or y_range < min_range:
        return True
    aspect = max(x_range, y_range) / max(min(x_range, y_range), 0.01)
    return aspect > max_aspect


def filter_spatially_distant_strokes(
    strokes: list[list[float]],
    tight_factor: float = 0.2,
    loose_factor: float = 0.6,
    dot_max_pts: int = 8,
) -> list[list[float]]:
    """Remove pen-down segments spatially far from the main character body.

    Main body is identified by largest bounding-box area (width × height), which is
    robust against long thin stray lines that have many points but near-zero height.
    Small segments (diacritical dots, ≤ dot_max_pts) get a generous loose_factor
    margin; larger segments get a tight_factor margin to exclude wandering strokes.
    """
    if not strokes:
        return strokes

    segments: list[tuple[int, int, list[list[float]]]] = []
    in_seg = False
    seg_start = 0
    seg_points: list[list[float]] = []

    for i, s in enumerate(strokes):
        pen_down = s[2] < 0.5
        if pen_down and not in_seg:
            seg_start = i
            in_seg = True
            seg_points = [[s[0], s[1]]]
        elif pen_down and in_seg:
            seg_points.append([s[0], s[1]])
        elif not pen_down and in_seg:
            segments.append((seg_start, i + 1, seg_points[:]))
            in_seg = False
            seg_points = []
    if in_seg:
        segments.append((seg_start, len(strokes), seg_points[:]))

    if len(segments) <= 1:
        return strokes

    # Main body = first substantial stroke (Arabic body is always drawn first).
    # Fall back to largest bounding-box area if the first stroke is tiny (< 10 pts).
    FIRST_STROKE_MIN_PTS = 10

    def _bbox_area(pts: list[list[float]]) -> float:
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        return max(max(xs) - min(xs), 1.0) * max(max(ys) - min(ys), 1.0)

    if len(segments[0][2]) >= FIRST_STROKE_MIN_PTS:
        main_pts = segments[0][2]
    else:
        main_pts = max(segments, key=lambda s: _bbox_area(s[2]))[2]
    main_xs = [p[0] for p in main_pts]
    main_ys = [p[1] for p in main_pts]
    min_x, max_x = min(main_xs), max(main_xs)
    min_y, max_y = min(main_ys), max(main_ys)

    dim = max(max_x - min_x, max_y - min_y, 10.0)
    tight_m = tight_factor * dim
    loose_m = loose_factor * dim

    result: list[list[float]] = []
    for seg_start, seg_end, pts in segments:
        cx = sum(p[0] for p in pts) / len(pts)
        cy = sum(p[1] for p in pts) / len(pts)
        margin = loose_m if len(pts) <= dot_max_pts else tight_m
        if (min_x - margin) <= cx <= (max_x + margin) and (min_y - margin) <= cy <= (max_y + margin):
            result.extend(strokes[seg_start:seg_end])

    return result if result else strokes


# Per-character dot configuration: (expected_dot_count, 'above' | 'below').
# Model coordinate space: y+ = up, so "above" = higher y, "below" = lower y.
_DOT_CONFIG: dict[str, tuple[int, str]] = {
    'ب': (1, 'below'),
    'ت': (2, 'above'),
    'ث': (3, 'above'),
    'ج': (1, 'below'),
    'خ': (1, 'above'),
    'ذ': (1, 'above'),
    'ز': (1, 'above'),
    'ش': (3, 'above'),
    'ض': (1, 'above'),
    'ظ': (1, 'above'),
    'غ': (1, 'above'),
    'ف': (1, 'above'),
    'ق': (2, 'above'),
    'ن': (1, 'above'),
    'ي': (2, 'below'),
}
_DOT_SEG_MAX_PTS: int = 10  # segments with ≤ this many points are treated as dots


def canonicalize_dot_positions(strokes: list[list[float]], char: str) -> list[list[float]]:
    """Place dot segments at canonical positions relative to the character body.

    Uses _DOT_CONFIG to determine expected dot count and direction per character.
    Dots are placed centered on the body horizontally:
      1 dot  → centered at body_center_x
      2 dots → side-by-side at body_center_x ± spacing
      3 dots → pyramid: 2 on bottom row ± spacing, 1 apex centered above/below them
    All dot segments are moved; their internal shape (relative offsets) is preserved.
    """
    char = char.strip()
    if char not in _DOT_CONFIG:
        return strokes
    dot_count, direction = _DOT_CONFIG[char]

    if not strokes:
        return strokes

    # Parse into (start_idx, end_idx, [xy points])
    segs: list[tuple[int, int, list[list[float]]]] = []
    in_seg = False
    seg_start = 0
    seg_pts: list[list[float]] = []
    for i, s in enumerate(strokes):
        pen_down = s[2] < 0.5
        if pen_down and not in_seg:
            seg_start = i
            in_seg = True
            seg_pts = [[s[0], s[1]]]
        elif pen_down and in_seg:
            seg_pts.append([s[0], s[1]])
        elif not pen_down and in_seg:
            segs.append((seg_start, i + 1, seg_pts[:]))
            in_seg = False
            seg_pts = []
    if in_seg:
        segs.append((seg_start, len(strokes), seg_pts[:]))

    if len(segs) < 2:
        return strokes  # no secondary segments to place

    def _bbox_area(pts: list[list[float]]) -> float:
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        return max(max(xs) - min(xs), 1.0) * max(max(ys) - min(ys), 1.0)

    # Body = first segment with >= 10 pts, else largest bounding-box segment
    if len(segs[0][2]) >= 10:
        body_idx = 0
    else:
        body_idx = max(range(len(segs)), key=lambda i: _bbox_area(segs[i][2]))

    body_pts = segs[body_idx][2]
    body_xs = [p[0] for p in body_pts]
    body_ys = [p[1] for p in body_pts]
    body_min_x, body_max_x = min(body_xs), max(body_xs)
    body_min_y, body_max_y = min(body_ys), max(body_ys)
    body_center_x = (body_min_x + body_max_x) / 2.0
    body_width = max(body_max_x - body_min_x, 1.0)
    body_height = max(body_max_y - body_min_y, 1.0)

    # Collect dot segments (small, not the body), sorted left-to-right by x centroid
    dot_segs_all = [
        (s, e, pts)
        for i, (s, e, pts) in enumerate(segs)
        if i != body_idx and len(pts) <= _DOT_SEG_MAX_PTS
    ]
    if not dot_segs_all:
        return strokes

    dot_segs_all.sort(key=lambda t: sum(p[0] for p in t[2]) / len(t[2]))
    # Keep only as many as expected
    dot_segs = dot_segs_all[:dot_count]

    # Geometry parameters
    spacing = max(body_width * 0.18, 0.35)   # horizontal gap between dots
    gap = max(body_height * 0.22, 0.28)       # gap from body edge to dot center

    if direction == 'above':
        base_y = body_max_y + gap
        pyramid_apex_y = base_y + spacing     # apex sits higher (larger y = up)
    else:
        base_y = body_min_y - gap
        pyramid_apex_y = base_y - spacing     # apex sits lower (smaller y = down)

    # Target (x, y) for each dot slot
    if dot_count == 1:
        targets = [(body_center_x, base_y)]
    elif dot_count == 2:
        targets = [
            (body_center_x - spacing, base_y),
            (body_center_x + spacing, base_y),
        ]
    else:  # 3 dots — pyramid
        targets = [
            (body_center_x - spacing, base_y),   # bottom-left
            (body_center_x + spacing, base_y),   # bottom-right
            (body_center_x, pyramid_apex_y),      # apex
        ]

    # Move each dot segment to its canonical position
    result = [list(s) for s in strokes]
    for i, (seg_start, seg_end, pts) in enumerate(dot_segs):
        if i >= len(targets):
            break
        target_x, target_y = targets[i]
        cx = sum(p[0] for p in pts) / len(pts)
        cy = sum(p[1] for p in pts) / len(pts)
        dx = target_x - cx
        dy = target_y - cy
        for j in range(seg_start, seg_end):
            result[j][0] += dx
            result[j][1] += dy

    return result


def trim_to_n_segments(strokes: list[list[float]], n: int) -> list[list[float]]:
    """Keep only the first n pen-down segments, dropping everything after."""
    count = 0
    in_seg = False
    for i, s in enumerate(strokes):
        if s[2] < 0.5:
            if not in_seg:
                count += 1
                in_seg = True
            if count > n:
                return strokes[:i]
        else:
            in_seg = False
    return strokes


def convert_offsets_to_absolute_coords(stroke_offsets: list[list[float]]) -> list[list[float]]:
    if not stroke_offsets:
        return []
    
    # convert to numpy for vectorized operations
    strokes_array = np.array(stroke_offsets)
    
    # vectorized cumulative sum for x and y 
    strokes_array[:, 0] = np.cumsum(strokes_array[:, 0])  # cumulative dx
    strokes_array[:, 1] = np.cumsum(strokes_array[:, 1])  # cumulative dy
    
    return strokes_array.tolist()