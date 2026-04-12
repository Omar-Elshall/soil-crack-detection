import { BrowserRouter, Routes, Route } from "react-router-dom";
import { Sidebar } from "./components/Sidebar";
import LivePage from "./pages/LivePage";
import HistoryPage from "./pages/HistoryPage";
import PostFlightPage from "./pages/PostFlightPage";

export default function App() {
  return (
    <BrowserRouter>
      <div className="h-screen flex bg-parchment overflow-hidden">
        <Sidebar />
        <main className="flex-1 flex flex-col overflow-hidden">
          <Routes>
            <Route path="/"               element={<LivePage />} />
            <Route path="/history"        element={<HistoryPage />} />
            <Route path="/missions/:id"   element={<PostFlightPage />} />
          </Routes>
        </main>
      </div>
    </BrowserRouter>
  );
}
