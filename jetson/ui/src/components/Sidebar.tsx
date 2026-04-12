import { NavLink } from "react-router-dom";
import { Radio, BookOpen, Map, Sun, Moon } from "lucide-react";
import { useTheme } from "../App";

const links = [
  { to: "/",        label: "Live",    icon: Radio    },
  { to: "/plan",    label: "Plan",    icon: Map      },
  { to: "/history", label: "History", icon: BookOpen },
];

export function Sidebar() {
  const { dark, toggle } = useTheme();

  return (
    <aside className="w-14 flex flex-col items-center py-4 gap-1 shrink-0"
      style={{ background: "#0D1117", borderRight: "1px solid #21262D" }}>

      {/* Logo mark */}
      <div className="w-8 h-8 rounded border flex items-center justify-center mb-4"
        style={{ borderColor: "rgb(var(--accent) / 0.4)" }}>
        <span className="font-display font-bold text-sm leading-none"
          style={{ color: "rgb(var(--accent))" }}>
          S
        </span>
      </div>

      {/* Nav links */}
      {links.map(({ to, label, icon: Icon }) => (
        <NavLink
          key={to}
          to={to}
          end
          title={label}
          className={({ isActive }) =>
            `w-10 h-10 rounded flex items-center justify-center transition-colors ${
              isActive
                ? "bg-white/10 text-white"
                : "text-white/30 hover:text-white/70 hover:bg-white/5"
            }`
          }
        >
          <Icon size={18} />
        </NavLink>
      ))}

      {/* Spacer */}
      <div className="flex-1" />

      {/* Theme toggle */}
      <button
        onClick={toggle}
        title={dark ? "Switch to light" : "Switch to dark"}
        className="w-10 h-10 rounded flex items-center justify-center text-white/30 hover:text-white/70 hover:bg-white/5 transition-colors"
      >
        {dark ? <Sun size={16} /> : <Moon size={16} />}
      </button>
    </aside>
  );
}
