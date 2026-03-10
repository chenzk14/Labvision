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

  // 多物体识别
  recognizeMultipleFile: (file, minConfidence = 0.5, topk = 5) => {
    const fd = new FormData()
    fd.append('file', file)
    fd.append('min_confidence', minConfidence)
    fd.append('topk', topk)
    return client.post('/api/recognize/multiple', fd).then(r => r.data)
  },
  recognizeMultipleBase64: (b64, minConfidence = 0.5, topk = 5) =>
    client.post('/api/recognize/multiple/base64', {
      image: b64,
      min_confidence: minConfidence,
      topk: topk
    }).then(r => r.data),

  // 日志
  getLogs: () => client.get('/api/logs').then(r => r.data),

  // 永久删除试剂
  deleteReagentPermanent: (reagentId) =>
    client.delete(`/api/reagents/${reagentId}/permanent`).then(r => r.data),
}