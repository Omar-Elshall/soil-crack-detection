import { NavLink } from "react-router-dom";
import { Radio, BookOpen } from "lucide-react";

const links = [
  { to: "/",        label: "Live",    icon: Radio    },
  { to: "/history", label: "History", icon: BookOpen },
];

export function Sidebar() {
  return (
    <aside className="w-14 bg-ink flex flex-col items-center py-4 gap-1 shrink-0">
      {/* Logo mark */}
      <div className="w-8 h-8 rounded border border-terracotta/40 flex items-center justify-center mb-4">
        <span className="text-terracotta font-display text-sm italic font-normal leading-none">S</span>
      </div>

      {links.map(({ to, label, icon: Icon }) => (
        <NavLink
          key={to}
          to={to}
          end
          title={label}
          className={({ isActive }) =>
            `w-10 h-10 rounded flex items-center justify-center transition-colors ${
              isActive
                ? "bg-terracotta/20 text-terracotta"
                : "text-ink-faint hover:text-parchment hover:bg-white/5"
            }`
          }
        >
          <Icon size={18} />
        </NavLink>
      ))}
    </aside>
  );
}
