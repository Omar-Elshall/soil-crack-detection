import { Area, AreaChart, ResponsiveContainer, Tooltip, YAxis } from "recharts";
import type { Detection } from "../hooks/useDetections";

interface Props { history: Detection[] }

export function CrackRatioChart({ history }: Props) {
  const data = history.map((d, i) => ({ i, v: d.crack_ratio_pct }));

  return (
    <div className="w-full h-20">
      <ResponsiveContainer width="100%" height="100%">
        <AreaChart data={data} margin={{ top: 4, right: 0, left: 0, bottom: 0 }}>
          <defs>
            <linearGradient id="crackGrad" x1="0" y1="0" x2="0" y2="1">
              <stop offset="5%"  stopColor="rgb(var(--accent))" stopOpacity={0.4} />
              <stop offset="95%" stopColor="rgb(var(--accent))" stopOpacity={0.03} />
            </linearGradient>
          </defs>
          <YAxis domain={[0, 100]} hide />
          <Tooltip
            content={({ active, payload }) =>
              active && payload?.length ? (
                <div className="bg-surface border border-parchment-darker rounded px-2 py-1 text-xs font-mono text-ink shadow-card">
                  {Number(payload[0].value).toFixed(1)}%
                </div>
              ) : null
            }
          />
          <Area
            type="monotone"
            dataKey="v"
            stroke="rgb(var(--accent))"
            strokeWidth={1.5}
            fill="url(#crackGrad)"
            dot={false}
            isAnimationActive={false}
          />
        </AreaChart>
      </ResponsiveContainer>
    </div>
  );
}
