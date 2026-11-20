/** Mirrors the tailwind.config that used to live in index.html <head>. */
module.exports = {
    content: ['../index.html'],
    theme: {
        extend: {
            fontFamily: {
                sans: ['Inter', 'ui-sans-serif', 'system-ui', '-apple-system', 'Segoe UI', 'Roboto', 'Helvetica Neue', 'Arial', 'sans-serif'],
                mono: ['IBM Plex Mono', 'ui-monospace', 'SFMono-Regular', 'Menlo', 'Consolas', 'Liberation Mono', 'monospace'],
            },
            colors: {
                obsidian: '#090d16',
                ink: '#0a0f1d',
            },
        },
    },
    plugins: [],
};
