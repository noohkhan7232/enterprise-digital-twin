# Security Policy

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 1.0.x   | :white_check_mark: |
| < 1.0   | :x:                |

Security fixes are applied to the latest 1.0.x release. Older pre-release
snapshots are not maintained.

## Reporting a Vulnerability

Please **do not** open a public issue for security vulnerabilities.

Report vulnerabilities privately using one of the following channels:

1. **GitHub private vulnerability reporting** (preferred):
   use the *"Report a vulnerability"* button under the repository's
   **Security** tab, which opens a private advisory visible only to the
   maintainers.
2. **Email:** nooh.khan840@gmail.com — include `[SECURITY]` in the subject
   line.

When reporting, please include where practical:

- A description of the vulnerability and its potential impact.
- Steps to reproduce, or a proof-of-concept.
- The affected component (module, script, container image, or manifest)
  and version/commit.
- Any suggested remediation, if you have one.

## Response Expectations

- **Acknowledgement** of your report within **72 hours**.
- An **initial assessment** (accepted, declined, or needs more information)
  within **7 days**.
- A remediation plan and, where applicable, a fixed release for confirmed
  vulnerabilities. Timelines depend on severity; critical issues are
  prioritised ahead of all other work.
- Credit in the release notes for responsibly disclosed findings, unless you
  prefer to remain anonymous.

## Responsible Disclosure

We ask reporters to:

- Give the maintainers a reasonable opportunity to remediate before any
  public disclosure (a **90-day** disclosure window is suggested).
- Avoid accessing, modifying, or destroying data that is not your own while
  investigating.
- Avoid actions that degrade service for others (e.g., denial-of-service
  testing) against any deployed instance you do not own.

In return, we commit to working with you in good faith, keeping you informed
of progress, and not pursuing action against good-faith security research
conducted within these guidelines.

## Scope Notes

- This repository ships a reference platform; deployments are self-hosted.
  Vulnerabilities in third-party dependencies should be reported upstream,
  though we still welcome a heads-up if a dependency issue affects this
  project's default configuration.
- Hardening guidance for the container image and Kubernetes manifests is
  documented in `week12_phase4_validation/security_review.md`.
