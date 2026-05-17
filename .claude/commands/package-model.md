Package the latest ML checkpoint into `ml/packaged_models/` so the API can serve it.

Look in `ml/outputs/` for the most recent checkpoint directory (highest step number). Then run:

```bash
cd ml && python -m src.package_model --pkg_name handwriting_model
```

After packaging, confirm that both `ml/packaged_models/handwriting_model.pt` and `ml/packaged_models/handwriting_model.scripted.pt` were updated (check modification time). Remind the user to restart the API to load the new model.
