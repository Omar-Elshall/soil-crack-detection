import { defineConfig, loadEnv } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, process.cwd(), 'VITE_')
  const jetson = env.VITE_INFERENCE_URL?.replace(/:\d+$/, '') ?? 'http://192.168.1.233'

  return {
    plugins: [react()],
    server: {
      proxy: {
        '/api/inference': {
          target: env.VITE_INFERENCE_URL ?? `${jetson}:8001`,
          rewrite: (p) => p.replace(/^\/api\/inference/, ''),
          changeOrigin: true,
        },
        '/api/mavlink': {
          target: env.VITE_MAVLINK_URL ?? `${jetson}:8002`,
          rewrite: (p) => p.replace(/^\/api\/mavlink/, ''),
          changeOrigin: true,
        },
        '/api/data': {
          target: env.VITE_DATA_URL ?? `${jetson}:8003`,
          rewrite: (p) => p.replace(/^\/api\/data/, ''),
          changeOrigin: true,
        },
      },
    },
  }
})
