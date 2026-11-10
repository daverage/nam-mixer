# NAM Mixer v0.5.8

- Dynamic Hybrid now shows soft, typical and hard input levels next to the crossover, with changeover boundaries and estimated time spent on each amp.
- Centre the crossover on the prepared performance, or audition input 6 dB quieter or hotter. Guidance explains input drive, output volume and matching your guitar to the preview.
- Expanded TONE3000 packs provide more room for files and chat at narrow panel widths.
- TONE3000 NAM downloads use OAuth account authorization and the returned model URL.
- Kaggle private-dataset verification allows five minutes for file indexing after upload.
- Journey-chart text alternatives and clearer screen-reader announcements improve accessibility.

Validation: all 50 frontend tests and JavaScript syntax checks pass. Python tests: 1,036 passed, 12 skipped, six failed. The same six failures reproduce on the pre-feature baseline: cloud-training fixture RF metadata (three cases), OAuth download fixtures (two), and an AI prompt-length assertion (one). Browser visual verification was unavailable. These existing failures remain unresolved in this release.

Installer build provenance: the published v0.5.8 tag is unchanged. The replacement installer workflow uses commit `7515683`, whose only changes after the tag are a macOS runner update from macOS 14 to macOS 15 and review progress notes. Application code and version match v0.5.8.
