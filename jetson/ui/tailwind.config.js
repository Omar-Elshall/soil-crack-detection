/** @type {import('tailwindcss').Config} */

// CSS variable color helper — uses CSS Color Level 4 slash syntax so
// opacity modifiers (bg-accent/20) and bare classes both produce valid CSS.
// rgba(R G B, A) is invalid; rgb(R G B / A) is the correct form.
const cv = (name) => `rgb(var(--${name}) / <alpha-value>)`;

export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  darkMode: "class",
  theme: {
    extend: {
      colors: {
        // Semantic tokens — values set via CSS variables in index.css
        // Light: cool slate-blue.  Dark: deep navy.
        parchment: {
          DEFAULT: cv("parchment"),
          dark:    cv("parchment-dark"),
          darker:  cv("parchment-darker"),
        },
        // Accent — Light: cyan-600  Dark: cyan-400
        terracotta: {
          DEFAULT: cv("accent"),
          light:   cv("accent-light"),
          dark:    cv("accent-dark"),
        },
        // Text — Light: slate-900→500  Dark: slate-100→500
        ink: {
          DEFAULT: cv("ink"),
          soft:    cv("ink-soft"),
          muted:   cv("ink-muted"),
          faint:   cv("ink-faint"),
        },
        moss: cv("positive"),
        sky:  cv("info"),
        // Card background — white in light, dark card in dark
        surface: cv("surface"),
        // Always-dark chrome (sidebar, camera bg, stop buttons)
        chrome: {
          DEFAULT: "#0D1117",
          soft:    "#161B22",
          border:  "#21262D",
        },
      },
      fontFamily: {
        display: ['"Outfit"', "system-ui", "sans-serif"],
        sans:    ['"Plus Jakarta Sans"', "system-ui", "sans-serif"],
        mono:    ['"Space Mono"', "monospace"],
      },
      borderRadius: {
        sm:      "2px",
        DEFAULT: "4px",
        md:      "6px",
        lg:      "10px",
        xl:      "14px",
      },
      borderColor: {
        DEFAULT: "rgb(var(--parchment-darker))",
      },
      boxShadow: {
        card: "0 1px 3px rgba(0,0,0,0.07), 0 1px 2px rgba(0,0,0,0.04)",
        "card-dark": "0 1px 3px rgba(0,0,0,0.4), 0 1px 2px rgba(0,0,0,0.3)",
      },
    },
  },
  plugins: [],
};
