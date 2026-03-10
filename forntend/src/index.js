// frontend/src/index.js
import React from 'react'
import ReactDOM from 'react-dom/client'
import { ConfigProvider } from 'antd'
import zhCN from 'antd/locale/zh_CN'
import App from './App'
import './styles/global.css'

const root = ReactDOM.createRoot(document.getElementById('root'))
root.render(
  <ConfigProvider
    locale={zhCN}
    theme={{
      token: {
        colorPrimary: '#1677ff',
        borderRadius: 6,
      }
    }}
  >
    <App />
  </ConfigProvider>
)