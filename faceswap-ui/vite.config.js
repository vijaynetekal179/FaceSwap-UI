import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import basicSsl from '@vitejs/plugin-basic-ssl'

export default defineConfig({
  plugins: [react(), basicSsl()],
  server: {
    host: true,
    proxy: {
      '/cdn-models': {
        target: 'https://d2vzalqc0smhzt.cloudfront.net',
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/cdn-models/, '')
      },
      '/api': {
        target: 'http://127.0.0.1:8000',
        changeOrigin: true
      }
    }
  }
})
