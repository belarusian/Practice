# Security

## Supported Use

This repository is a development and learning lab, not a production-hardened deployment.

The Docker Compose stack and Sunny ops scripts are intended for local or private lab networks. Review credentials, exposed ports, firewall rules, and cloud security groups before adapting them to any public environment.

## Reporting Issues

If you find a security issue, do not open a public issue with exploit details. Contact the maintainers privately through the repository owner profile or the security contact documented by the public repository once published.

## Secrets

Do not commit:

- cloud credentials
- SSH keys
- `.env` files with real values
- private hostnames, usernames, or IP addresses
- model weights or proprietary datasets

Use the provided `*.example` files as templates and keep private values in ignored local files.
