# molforge editor — frontend

React + React Flow. Builds into `src/molforge/gui/static/`, which the Python
server mounts, so a wheel ships the editor and a user never installs Node.

```sh
npm install
npm run build     # type-check, then build into the package
npm run dev       # dev server, proxying /api to molforge-gui on :8765
```

The build output is **not committed** — it's generated at release time. A
source checkout therefore has no editor until you run the build, and the
server serves a page saying exactly that rather than a 404.

## Where the shapes come from

Nothing here hard-codes which engines exist. `/api/blocks` returns the
catalog molforge derives from its own engine registry, and `src/api.ts`
mirrors those payload shapes as TypeScript interfaces — so a backend rename
surfaces as a build error rather than as an `undefined` at runtime.

## Port colours

`src/theme.css` holds the categorical palette; `src/ports.ts` maps port
types onto it. Nine port types share seven hues: `structure_ensemble`
shares `structure`'s and `designs` shares `sequence`'s, because "one versus
many" is a secondary encoding (the doubled ring) rather than a separate
identity. That keeps the palette inside its validated slots and leaves red
free to mean *invalid*.

Colour is never the only channel — every port shows its name, the inspector
names the type in words, and mismatched connections are refused outright.
