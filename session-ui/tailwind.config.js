// Build config for the vendored CSS (static/tailwind.css). Regenerate after
// changing any class names in static/index.html:
//   npx tailwindcss@3.4 -c session-ui/tailwind.config.js \
//       -i session-ui/tailwind.input.css -o session-ui/static/tailwind.css --minify
module.exports = {
  // 'class' means dark mode is switched by a `dark` class on <html>, which the SPA
  // toggles itself (theme button + saved preference) instead of following the OS.
  darkMode: 'class',
  content: ['./session-ui/static/index.html'],
  theme: { extend: {} },
  plugins: [],
}
