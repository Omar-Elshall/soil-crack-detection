import { useCallback, useEffect, useState } from "react";
import { listMissions, startMission, stopMission, type MissionMeta } from "../api/missions";

export function useMissions() {
  const [missions, setMissions] = useState<MissionMeta[]>([]);
  const [activeMissionId, setActiveMissionId] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  const refresh = useCallback(async () => {
    try {
      const list = await listMissions();
      setMissions(list);
      const active = list.find((m) => m.status === "active");
      if (active) setActiveMissionId(active.id);
    } catch { /* ignore */ }
  }, []);

  useEffect(() => { refresh(); }, [refresh]);

  const start = useCallback(async () => {
    setLoading(true);
    try {
      const { mission_id } = await startMission();
      setActiveMissionId(mission_id);
      await refresh();
    } finally {
      setLoading(false);
    }
  }, [refresh]);

  const stop = useCallback(async () => {
    if (!activeMissionId) return;
    setLoading(true);
    try {
      await stopMission(activeMissionId);
      setActiveMissionId(null);
      await refresh();
    } finally {
      setLoading(false);
    }
  }, [activeMissionId, refresh]);

  return { missions, activeMissionId, loading, start, stop, refresh };
}
