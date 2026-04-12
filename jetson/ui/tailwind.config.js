/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        parchment: {
          DEFAULT: "#F9F6F1",
          dark: "#EEE8DE",
          darker: "#E0D8CC",
        },
        terracotta: {
          DEFAULT: "#C4622D",
          light: "#D97A47",
          dark: "#A04E22",
        },
        ink: {
          DEFAULT: "#2C2C2C",
          soft: "#4A4A4A",
          muted: "#7A7A7A",
          faint: "#AAAAAA",
        },
        moss: "#4E6E4E",
        sky: "#2B6CB0",
      },
      fontFamily: {
        display: ['"Instrument Serif"', "Georgia", "serif"],
        sans: ['"DM Sans"', "system-ui", "sans-serif"],
        mono: ['"JetBrains Mono"', "monospace"],
      },
      borderRadius: {
        sm: "2px",
        DEFAULT: "4px",
        md: "6px",
        lg: "10px",
      },
    },
  },
  plugins: [],
};

