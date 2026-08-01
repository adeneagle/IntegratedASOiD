// app.js — Which application this build is.
//
// One of the two files a host supplies to the shared labeling shell; the other
// is plugins/pipeline.js. Everything else — index.html, styles.css, core/,
// main.js, preload.js — is identical across builds, so this is the only place
// the A-SOID name appears in the UI layer.
//
// index.html carries placeholder copy marked data-app="name|tagline|gate",
// which core/state.js fills in from here.

window.LABELER_APP = {
  id: "asoid-desktop",

  // Distinct settings slots keep two labelers installed side by side from
  // overwriting each other's preferences.
  settingsKey: "asoidDesktop_settings",

  name: "A-SOID Desktop",
  tagline: "Label, extract pose features, train, and refine animal behavior classifiers.",
  gate: "A-SOID Desktop keeps your videos, pose files, labels, features, and models together in a project.",
};
