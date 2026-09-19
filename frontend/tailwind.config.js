/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        qgis: {
          bg: "#F5F5F5",
          panel: "#FFFFFF",
          border: "#E0E0E0",
          header: "#EAEAEA",
          hover: "#E4E4E4",
          status: "#ECECEC",
          text: "#222222",
          muted: "#666666",
        }
      },
      fontSize: {
        xxs: "11px",
      }
    },
  },
  plugins: [],
}
