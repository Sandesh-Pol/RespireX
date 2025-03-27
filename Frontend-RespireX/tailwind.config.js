/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors:{
        "primary-blue":"#3c83ff",
        "primary-gray":"#3d4b56",
        "dark-blue":"#20304A"
      }
    },
  },
  plugins: [],
}