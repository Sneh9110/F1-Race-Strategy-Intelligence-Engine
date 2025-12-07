/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        // F1 Team Colors
        'redbull': {
          DEFAULT: '#0600EF',
          dark: '#120078',
          light: '#1E1EF0',
        },
        'ferrari': {
          DEFAULT: '#DC0000',
          dark: '#8B0000',
          light: '#FF2800',
        },
        'mercedes': {
          DEFAULT: '#00D2BE',
          dark: '#008B80',
          light: '#27F4E2',
        },
        'mclaren': {
          DEFAULT: '#FF8700',
          dark: '#D66400',
          light: '#FFA230',
        },
        // Traffic Light System
        'traffic': {
          green: '#10B981',
          amber: '#F59E0B',
          red: '#EF4444',
        },
        // Racing Theme
        'racing': {
          black: '#15151E',
          darkgray: '#1F1F2E',
          gray: '#2D2D3F',
          lightgray: '#4A4A5E',
          white: '#F8F9FA',
        },
      },
      fontFamily: {
        'formula': ['Titillium Web', 'sans-serif'],
      },
      animation: {
        'pulse-slow': 'pulse 3s cubic-bezier(0.4, 0, 0.6, 1) infinite',
        'spin-slow': 'spin 3s linear infinite',
      },
    },
  },
  plugins: [],
}
