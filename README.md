# minidump-diagnostics-tool

# Requirements

### Redis:
- I hosted redis locally on Docker Desktop, you can either setup locally (Docker suggested) or run it through a cloud provider (Redis lets you host for free)
```bash
> docker pull redis/redis-stack:latest
> docker run -d --name redis-stack -p 6379:6379 -p 8001:8001 
> redis/redis-stack:latest
```

