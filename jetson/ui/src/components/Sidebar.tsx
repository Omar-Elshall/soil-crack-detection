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
    <aside
      className="w-14 flex flex-col items-center py-4 gap-1 shrink-0 border-r border-parchment-darker bg-parchment-dark"
    >
      {/* Logo mark */}
      <div
        className="w-8 h-8 rounded-md flex items-center justify-center mb-4 bg-terracotta/15 border border-terracotta/30"
      >
        <span className="font-display font-bold text-sm leading-none text-terracotta">S</span>
      </div>

      {/* Nav links */}
      {links.map(({ to, label, icon: Icon }) => (
        <NavLink
          key={to}
          to={to}
          end
          title={label}
          className={({ isActive }) =>
            `w-10 h-10 rounded-md flex items-center justify-center transition-colors ${
              isActive
                ? "bg-terracotta/15 text-terracotta"
                : "text-ink-muted hover:text-ink hover:bg-parchment-darker/60"
            }`
          }
        >
          <Icon size={18} />
        </NavLink>
      ))}

      <div className="flex-1" />

      {/* Theme toggle */}
      <button
        onClick={toggle}
        title={dark ? "Switch to light" : "Switch to dark"}
        className="w-10 h-10 rounded-md flex items-center justify-center text-ink-muted hover:text-ink hover:bg-parchment-darker/60 transition-colors"
      >
        {dark ? <Sun size={16} /> : <Moon size={16} />}
      </button>
    </aside>
  );
}
