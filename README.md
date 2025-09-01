# minidump-diagnostics-tool

- View Diagram @ https://lucid.app/lucidchart/b2a4e86e-0c9c-49bd-9f93-da2d936a896b/edit?viewport_loc=-1894%2C-2592%2C5726%2C2099%2C0_0&invitationId=inv_bf505244-487b-4760-b76a-aeca78b58b1d


# Requirements

### Redis:
- I hosted redis locally on Docker Desktop, you can either setup locally (Docker suggested) or run it through a cloud provider (Redis lets you host for free)
```bash
> docker pull redis/redis-stack:latest
> docker run -d --name redis-stack -p 6379:6379 -p 8001:8001 
> redis/redis-stack:latest
```

