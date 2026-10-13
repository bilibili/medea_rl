# CASTER project website

A bilingual static research website for CASTER, inspired by LAIT’s warm editorial style. All assets are local; there is no build step or package dependency.

Project URL: https://bilibili.github.io/medea_rl/

## Local preview

From the repository root:

```sh
python3 -m http.server 8794 --bind 127.0.0.1 --directory website
```

Visit http://127.0.0.1:8794. A local server enables video subtitles and clipboard support.

## Deployment

GitHub Pages must use **GitHub Actions** as its source in the repository’s Settings → Pages. The `Deploy CASTER website` workflow publishes only this directory on pushes to `main` affecting `website/` or the workflow itself. It can also be run manually from Actions. If Pages is enabled after the first deployment attempt, re-run the workflow once to publish the existing site.

`index.html` uses relative asset paths and absolute canonical / Open Graph URLs for the project URL above. If the site moves, update those URLs.

## Editing

- `index.html`: layout and static English / Chinese copy.
- `styles.css`: design and responsive layouts.
- `app.js`: viewer perspectives, training stages, result values, and interactions.
- `assets/`: figures, video, subtitles, paper PDF, and BibTeX.
- `SOURCES.md`: public research sources and metric definitions.

Viewer quotations are model-generated reactions from the paper, not real audience testimonials. Chinese quotations are translations. Project media remain the property of their respective rights holders.
