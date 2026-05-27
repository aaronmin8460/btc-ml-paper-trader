/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{ts,tsx}'],
  theme: {
    extend: {
      colors: {
        btc: {
          400: '#f8b84e',
          500: '#f7931a',
          600: '#d97706',
        },
        ink: {
          950: '#05070d',
          900: '#0a0f1c',
          800: '#111827',
        },
      },
      boxShadow: {
        glow: '0 0 40px rgba(247, 147, 26, 0.16)',
      },
      fontFamily: {
        sans: ['Inter', 'ui-sans-serif', 'system-ui', 'sans-serif'],
      },
    },
  },
  plugins: [],
};
