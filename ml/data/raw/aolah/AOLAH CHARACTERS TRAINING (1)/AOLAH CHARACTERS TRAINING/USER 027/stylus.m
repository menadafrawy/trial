function stylus
% Pen-Like data processing template
% pen.m is a GUI ready to use
%       the GUI calls a function called "strokes"
%       located at the end of the file every time the pen passes over the red area
%       

global Press_State str str1 str2
s1 = input(['Please enter a name for the sample stroke file from the following list' , ...
'\n [ alif , ba , ta , tha , jeem , hha , kha , dal , thal , ra  , zay , seen , sheen , sad ,' , ...
'\n  dhad , tta ,zha , ain , ghain , fa , qaf , kaf , lam , meem , noon , ha , waw , ya ]' , ...
'\n ' , ...
'\n '],'s');
s21 = '.csv'; s22 = strcat(s1,s21); 
s31 = ' st.csv'; s32 = strcat(s1, s31);
s41 = ' strokes.csv'; s42 = strcat(s1,s41);
str11 = strcat(s1,' PreStro'); str12 = strcat(s1,' Strokes');
mkdir(str11); mkdir(str12);

str = strcat(str11,'\',s22);
str1 = strcat(str11,'\',s32);
str2 = strcat(str12,'\',s42);

himatge = findobj('tag','PEN');
if (isempty(himatge))
   Press_State = 0;

    % create the new figure
    himatge = figure; % a figure is generated and assigned an identifie
	
	set(himatge,'numbertitle','off');                   % remove the figure number
    set(himatge,'name','STYLUS PEN SIMULATOR');         % name
    set(himatge,'MenuBar','none');                      % remove the icon menu
    set(himatge,'doublebuffer','on');                   % two buffers of graphics
    set(himatge,'tag','PEN');                           % identify the figure
    set(himatge,'Color',[0.95 0.95 0.95]);
    set(himatge,'WindowButtonMotionFcn',@moviment);
    set(himatge,'WindowButtonDownFcn',@moviment_down);
    set(himatge,'WindowButtonUpFcn',@moviment_up);
       
    % create the axis
    h_axes = axes('position', [0 0 1 0.85]);
    set(h_axes,'Tag','AXES');
    box(h_axes,'on');
    grid(h_axes,'on');
    axis(h_axes,[0 1 0 1]);
    axis(h_axes,'off');
    hold(h_axes,'on');
    
    % create eraser marks
    fill([0 0.1 0.1 0 0],[0 0 0.1 0.1 0],'r');
    fill([0.9 1 1 0.9 0.9],[0 0 0.1 0.1 0],'r');
    
    % create the text
    h_text = uicontrol('Style','edit','Units','normalized','Position',[0 0.85 1 0.15],'FontSize',14,'HorizontalAlignment','left','Enable','inactive','Tag','TEXT');
       
	% ######  MENU  ######################################
    h_opt = uimenu('Label','&Options');
        uimenu(h_opt,'Label','Manual mode {input processed at the red area}','Enable','off');
        uimenu(h_opt,'Label','Clear / Restore','Callback',@show);
        h_ref = uimenu(h_opt,'Label','References','separator','on');
        uimenu(h_ref,'Label','Rectangular','Callback',@referencies,'Tag','NORMAL','UserData','REF');
        uimenu(h_ref,'Label','Italic','Callback',@referencies,'Tag','ITALIC','UserData','REF');
        uimenu(h_ref,'Label','None','Callback',@referencies,'Tag','CAP','UserData','REF','Checked','on');
    
        uimenu(h_opt,'Label','Sortir - Exit','Callback','closereq;','separator','on');
        
    h_opt2 = uimenu('Label','&About pen.m');
        uimenu(h_opt2,'Label','Grup de Robotica de la Universitat de Lleida');
        uimenu(h_opt2,'Label','Grupo de Robotica de la Universidad de Lleida');
        uimenu(h_opt2,'Label','Robotic Team, University of Lleida (Spain)');
        uimenu(h_opt2,'Label','Browse -> http://robotica.udl.cat','Callback','web http://robotica.udl.cat -browser','separator','on');
        
else
    figure(himatge);
end

% #########################################################################

% #########################################################################
function show(hco,eventStruct)
% select mode
global x_pen y_pen str1

% delete previous drawing
delete(findobj('Tag','TRAJECTORIA'));
delete(findobj('Tag','TRAJECTORIA2'));

% delete previous data
x_pen = [];
y_pen = [];

delete(str1);
% if yes
himatge = findobj('tag','PEN');

set(himatge,'WindowButtonMotionFcn',@moviment);
set(himatge,'WindowButtonDownFcn',@moviment_down);
set(himatge,'WindowButtonUpFcn',@moviment_up);
% #########################################################################

% #########################################################################
function referencies(hco,eventStruct)
% mark referrals

set(findobj('UserData','REF'),'Checked','off');
set(findobj('Tag',get(hco,'Tag')),'Checked','on');

% delete previous
delete(findobj('Tag','REFERENCIES'));

% draw new ones
h_axes = findobj('Tag','AXES');

switch get(hco,'Tag')
case {'NORMAL'}
    for i = 0:0.1:1
        % horizontal
        plot(h_axes,[0 1],[i i],'k:','Color',[0.5 0.5 0.5],'Tag','REFERENCIES');
        
        % vertical
        plot(h_axes,[i i],[0 1],'k:','Color',[0.5 0.5 0.5],'Tag','REFERENCIES');
    end
    
case {'ITALIC'}
    for i = 0:0.1:1
        % horitzontal
        plot(h_axes,[0 1],[i i],'k:','Color',[0.5 0.5 0.5],'Tag','REFERENCIES');
        
        % vertical
        plot(h_axes,[i i+0.1],[0 1],'k:','Color',[0.5 0.5 0.5],'Tag','REFERENCIES');
    end
end
% #########################################################################

% #########################################################################
function moviment_down(hco,eventStruct)
% ensure that there is no catch in progress
global Press_State x_pen y_pen str1

% toggle
Press_State = 1;

% retrieve point
h_axes = findobj('Tag','AXES');
p = get(h_axes,'CurrentPoint');
x = p(1,1);
y = p(1,2);
dlmwrite(str1,p,'precision','%10.10f','delimiter',',','-append');

% cumulative data
x_pen = [x_pen x];
y_pen = [y_pen y];


% draw
plot(h_axes,x,y,'b.','Tag','TRAJECTORIA');
% #########################################################################

% #########################################################################
function moviment_up(hco,eventStruct)
% ensure that there is no catch in progress
global Press_State x_pen y_pen

% toggle
Press_State = 0;

h_axes = findobj('Tag','AXES');

% analysis of what has been done
% erase previous box
delete(findobj('Tag','TRAJECTORIA2'));

% mark a box
x_i = min(x_pen);
x_f = max(x_pen);
x_d = max([1 (x_f - x_i)]);
y_i = min(y_pen);
y_f = max(y_pen);
y_d = max([1 (y_f - y_i)]);
plot(h_axes,[x_i x_f x_f x_i x_i],[y_i y_i y_f y_f y_i],'K:','MarkerSize',22,'Tag','TRAJECTORIA2');
% #########################################################################

% #########################################################################
function moviment(hco,eventStruct)
% ensure that there is no catch in progress
global Press_State x_pen y_pen

h_axes = findobj('Tag','AXES');

p = get(h_axes,'CurrentPoint');
x = p(1,1);
y = p(1,2);

if Press_State == 1
    % clicked button
    
    if ((y < 0) || (y > 1) || (x < 0) || (x > 1))
        % do nothing
        return;
    end
    
    if ((x ~= x_pen(end)) || (y ~= y_pen(end)))
        % next point
        
        x_pen = [x_pen x];
        y_pen = [y_pen y];

        plot(h_axes,[x_pen(end-1) x],[y_pen(end-1) y],'b.-','Tag','TRAJECTORIA');
        
    end
    
end


if ((x < 0.1) && (y < 0.1)) || ((x > 0.9) && (y < 0.1))
        % when passing over the red zone
        if (~isempty(x_pen))
            
            % get the strokes of the character
            strokes(x_pen,y_pen);
            
            % delete drawing
            delete(findobj('Tag','TRAJECTORIA'));
            delete(findobj('Tag','TRAJECTORIA2'));

            % delete data
            x_pen = [];
            y_pen = [];
            
            close all;
            
        end
end   

% #########################################################################

% #########################################################################

% #########################################################################
function strokes(x_pen,y_pen)
global str str1 str2

XY1 = csvread(str1);
PPx=XY1(:,1); PPy=XY1(:,2);
PPx1 = [];
for i = 1 : 2 : length(PPx)
    PPx1 = [PPx1 PPx(i)];
end
PPx1 = [PPx1,0];
PPy1 = [];
for i = 1 : 2 : length(PPy)
    PPy1 = [PPy1 PPy(i)];
end
PPy1 = [PPy1,0];
dlmwrite(str,[x_pen' y_pen'],'precision','%10.10f','delimiter',',','-append');
XY2 = csvread(str);
PPx2=XY2(:,1); PPy2=XY2(:,2);
k = 0;
for i = 1: length(PPx2)
    if PPx2(i)== PPx1(k+1)
        if PPy2(i)== PPy1(k+1)
            k=k+1;
        end
    end
    PPx3(i,1) = PPx2(i);
    PPy3(i,1) = PPy2(i);
    PPy3(i,2) = k;
end

dlmwrite(str2,[PPx3 PPy3],'precision','%10.10f','delimiter',',','-append');
