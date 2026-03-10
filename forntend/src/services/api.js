// frontend/src/services/api.js
import axios from 'axios'

const BASE_URL = 'http://localhost:8000'

const client = axios.create({
  baseURL: BASE_URL,
  timeout: 10000,
})

export const api = {
  // 系统状态
  getStatus: () => client.get('/api/status').then(r => r.data),

  // 试剂管理
  createReagent: (data) => client.post('/api/reagents', data).then(r => r.data),
  listReagents: () => client.get('/api/reagents').then(r => r.data),
  getReagent: (id) => client.get(`/api/reagents/${id}`).then(r => r.data),
  deleteReagent: (id) => client.delete(`/api/reagents/${id}`).then(r => r.data),

  // 图片注册
  registerImage: (reagentId, formData) =>
    client.post(`/api/reagents/${reagentId}/register-image`, formData, {
      headers: { 'Content-Type': 'multipart/form-data' }
    }).then(r => r.data),

  // 识别
  recognizeFile: (file) => {
    const fd = new FormData()
    fd.append('file', file)
    return client.post('/api/recognize', fd).then(r => r.data)
  },
  recognizeBase64: (b64) =>
    client.post('/api/recognize/base64', { image: b64 }).then(r => r.data),

  // 日志
  getLogs: () => client.get('/api/logs').then(r => r.data),
}