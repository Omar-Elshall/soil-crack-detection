import { BrowserRouter, Routes, Route } from "react-router-dom";
import { createContext, useContext, useEffect, useState } from "react";
import { Sidebar } from "./components/Sidebar";
import LivePage from "./pages/LivePage";
import HistoryPage from "./pages/HistoryPage";
import PostFlightPage from "./pages/PostFlightPage";

// ── Theme context ────────────────────────────────────────────────────────────
interface ThemeCtx { dark: boolean; toggle: () => void }
export const ThemeContext = createContext<ThemeCtx>({ dark: false, toggle: () => {} });
export const useTheme = () => useContext(ThemeContext);

export default function App() {
  const [dark, setDark] = useState<boolean>(() => {
    const stored = localStorage.getItem("theme");
    if (stored) return stored === "dark";
    return window.matchMedia("(prefers-color-scheme: dark)").matches;
  });

  useEffect(() => {
    document.documentElement.classList.toggle("dark", dark);
    localStorage.setItem("theme", dark ? "dark" : "light");
  }, [dark]);

  const toggle = () => setDark(d => !d);

  return (
    <ThemeContext.Provider value={{ dark, toggle }}>
      <BrowserRouter>
        <div className="h-screen flex bg-parchment overflow-hidden">
          <Sidebar />
          <main className="flex-1 flex flex-col overflow-hidden">
            <Routes>
              <Route path="/"             element={<LivePage />} />
              <Route path="/history"      element={<HistoryPage />} />
              <Route path="/missions/:id" element={<PostFlightPage />} />
            </Routes>
          </main>
        </div>
      </BrowserRouter>
    </ThemeContext.Provider>
  );
}
