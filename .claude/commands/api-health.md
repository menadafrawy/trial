Check whether the local API is running and healthy.

```bash
curl -s http://localhost:8000/health
```

Parse the response and report: status, model_loaded, device, and any missing resources. If the request fails entirely, the API is not running — remind the user to start it with `cd api/src && python main.py`.
