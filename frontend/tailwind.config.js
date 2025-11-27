/** @type {import('tailwindcss').Config} */
module.exports = {
  content: ["./src/**/*.{js,jsx,ts,tsx}"],
  theme: {
    extend: {
      colors: {
        background: "#0f172a", // Slate 900
        surface: "#1e293b", // Slate 800
        primary: {
          DEFAULT: "#2563eb", // Blue 600
          hover: "#1d4ed8", // Blue 700
          foreground: "#ffffff",
        },
        secondary: {
          DEFAULT: "#9333ea", // Purple 600
          hover: "#7e22ce", // Purple 700
          foreground: "#ffffff",
        },
        accent: {
          DEFAULT: "#8b5cf6", // Violet 500
          hover: "#7c3aed", // Violet 600
        },
        muted: {
          DEFAULT: "#334155", // Slate 700
          foreground: "#94a3b8", // Slate 400
        },
        border: "#1e293b", // Slate 800
      },
      fontFamily: {
        sans: ["Inter", "sans-serif"],
        heading: ["Outfit", "sans-serif"],
      },
      animation: {
        "fade-in": "fadeIn 0.5s ease-out",
        "slide-up": "slideUp 0.5s ease-out",
        "pulse-slow": "pulse 3s cubic-bezier(0.4, 0, 0.6, 1) infinite",
        "scale-in": "scaleIn 0.3s ease-out",
      },
      keyframes: {
        fadeIn: {
          "0%": { opacity: "0" },
          "100%": { opacity: "1" },
        },
        slideUp: {
          "0%": { transform: "translateY(20px)", opacity: "0" },
          "100%": { transform: "translateY(0)", opacity: "1" },
        },
        scaleIn: {
          "0%": { transform: "scale(0.9)", opacity: "0" },
          "100%": { transform: "scale(1)", opacity: "1" },
        },
      },
      backgroundImage: {
        "gradient-radial": "radial-gradient(var(--tw-gradient-stops))",
        "hero-pattern": "url('/public/hero-bg.svg')", // Placeholder if we add one
      },
    },
  },
  plugins: [],
};
