// Build config for the vendored CSS (static/tailwind.css). Regenerate after
// changing any class names in static/index.html:
//   npx tailwindcss@3.4 -c session-ui/tailwind.config.js \
//       -i session-ui/tailwind.input.css -o session-ui/static/tailwind.css --minify
module.exports = {
  darkMode: 'class',
  content: ['./session-ui/static/index.html'],
  theme: { extend: {} },
  plugins: [],
}
