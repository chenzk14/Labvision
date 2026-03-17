// frontend/src/App.jsx
import React, { useState } from 'react'
import { Layout, Menu, theme } from 'antd'
import {
  ExperimentOutlined,
  PlusCircleOutlined,
  SearchOutlined,
  DatabaseOutlined,
  LineChartOutlined,
  AppstoreOutlined,
  CheckCircleOutlined,
} from '@ant-design/icons'
import ReagentRegister from './pages/ReagentRegister'
// import ReagentRecognize from './pages/ReagentRecognize'
import MultipleRecognize from './pages/MultipleRecognize'
import ReagentList from './pages/ReagentList'
import Dashboard from './pages/Dashboard'
import CorrectionManage from './pages/CorrectionManage'
import './styles/global.css'

const { Header, Sider, Content } = Layout

const menuItems = [
  { key: 'dashboard', icon: <LineChartOutlined />, label: '系统概览' },
  { key: 'register', icon: <PlusCircleOutlined />, label: '试剂录入' },
  // { key: 'recognize', icon: <SearchOutlined />, label: '单试剂识别' },
  { key: 'multiple', icon: <AppstoreOutlined />, label: '多试剂识别' },
  { key: 'correction', icon: <CheckCircleOutlined />, label: '纠错管理' },
  { key: 'list', icon: <DatabaseOutlined />, label: '试剂库' },
]

const pageMap = {
  dashboard: <Dashboard />,
  register: <ReagentRegister />,
  // recognize: <ReagentRecognize />,
  multiple: <MultipleRecognize />,
  list: <ReagentList />,
  correction: <CorrectionManage />,
}

export default function App() {
  const [currentPage, setCurrentPage] = useState('dashboard')
  const { token } = theme.useToken()

  return (
    <Layout style={{ minHeight: '100vh' }}>
      <Sider
        width={220}
        style={{
          background: '#001529',
          boxShadow: '2px 0 8px rgba(0,0,0,0.3)',
        }}
      >
        <div style={{
          height: 64,
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          color: '#fff',
          fontSize: 16,
          fontWeight: 700,
          borderBottom: '1px solid rgba(255,255,255,0.1)',
        }}>
          <ExperimentOutlined style={{ marginRight: 8, color: '#52c41a' }} />
          试剂视觉系统
        </div>
        <Menu
          theme="dark"
          mode="inline"
          selectedKeys={[currentPage]}
          items={menuItems}
          onClick={({ key }) => setCurrentPage(key)}
          style={{ marginTop: 8 }}
        />
      </Sider>

      <Layout>
        <Header style={{
          background: token.colorBgContainer,
          padding: '0 24px',
          display: 'flex',
          alignItems: 'center',
          boxShadow: '0 1px 4px rgba(0,0,0,0.1)',
        }}>
          <span style={{ fontSize: 18, fontWeight: 600, color: token.colorText }}>
            {menuItems.find(m => m.key === currentPage)?.label}
          </span>
        </Header>

        <Content style={{
          margin: '24px',
          padding: '24px',
          background: token.colorBgContainer,
          borderRadius: token.borderRadiusLG,
          minHeight: 'calc(100vh - 64px - 48px)',
        }}>
          {pageMap[currentPage]}
        </Content>
      </Layout>
    </Layout>
  )
}