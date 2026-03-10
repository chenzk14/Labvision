
// frontend/src/pages/Dashboard.jsx
import React, { useState, useEffect } from 'react'
import { Card, Row, Col, Statistic, Alert, Tag, Spin, Button, Space } from 'antd'
import {
  ExperimentOutlined, DatabaseOutlined, ThunderboltOutlined, CheckCircleOutlined, ReloadOutlined
} from '@ant-design/icons'
import { api } from '../services/api'

export default function Dashboard() {
  const [status, setStatus] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)
  const [refreshing, setRefreshing] = useState(false)

  const loadStatus = async () => {
    setRefreshing(true)
    try {
      const data = await api.getStatus()
      setStatus(data)
      setError(null)
    } catch (e) {
      setError('无法连接后端服务，请确认已启动 uvicorn')
    } finally {
      setLoading(false)
      setRefreshing(false)
    }
  }

  useEffect(() => {
    loadStatus()
  }, [])

  // 页面获得焦点时自动刷新
  useEffect(() => {
    const handleVisibilityChange = () => {
      if (!document.hidden) {
        loadStatus()
      }
    }
    document.addEventListener('visibilitychange', handleVisibilityChange)
    return () => document.removeEventListener('visibilitychange', handleVisibilityChange)
  }, [])

  if (loading) return <div style={{ textAlign: 'center', padding: 80 }}><Spin size="large" /></div>

  if (error) return (
    <Alert
      type="error"
      message="连接失败"
      description={error}
      showIcon
      action={
        <Button type="primary" onClick={loadStatus}>
          重试连接
        </Button>
      }
      style={{ maxWidth: 600 }}
    />
  )

  return (
    <div>
      <div style={{ marginBottom: 16, textAlign: 'right' }}>
        <Space>
          <span style={{ color: '#666' }}>最后更新: {new Date().toLocaleTimeString()}</span>
          <Button 
            icon={<ReloadOutlined />} 
            onClick={loadStatus}
            loading={refreshing}
          >
            刷新数据
          </Button>
        </Space>
      </div>
      <Row gutter={16} style={{ marginBottom: 24 }}>
        <Col span={6}>
          <Card>
            <Statistic
              title="已注册试剂"
              value={status?.unique_reagent_ids || 0}
              prefix={<ExperimentOutlined />}
              valueStyle={{ color: '#1890ff' }}
              suffix="种"
            />
          </Card>
        </Col>
        <Col span={6}>
          <Card>
            <Statistic
              title="图片向量数"
              value={status?.faiss_vectors || 0}
              prefix={<DatabaseOutlined />}
              valueStyle={{ color: '#52c41a' }}
              suffix="张"
            />
          </Card>
        </Col>
        <Col span={6}>
          <Card>
            <Statistic
              title="注册记录"
              value={status?.total_registrations || 0}
              prefix={<CheckCircleOutlined />}
              valueStyle={{ color: '#faad14' }}
            />
          </Card>
        </Col>
        <Col span={6}>
          <Card>
            <Statistic
              title="推理设备"
              value={status?.device?.toUpperCase() || 'N/A'}
              prefix={<ThunderboltOutlined />}
              valueStyle={{ color: status?.device === 'cuda' ? '#52c41a' : '#ff7875' }}
            />
          </Card>
        </Col>
      </Row>

      <Card title="系统信息">
        <Row gutter={16}>
          <Col span={8}>
            <p><strong>模型骨干：</strong>{status?.model || 'N/A'}</p>
            <p><strong>推理设备：</strong>
              <Tag color={status?.device === 'cuda' ? 'green' : 'orange'}>
                {status?.device?.toUpperCase()}
              </Tag>
            </p>
          </Col>
          <Col span={8}>
            <p><strong>试剂种类：</strong>{status?.unique_reagent_names} 种</p>
            <p><strong>服务状态：</strong>
              <Tag color="green">运行中</Tag>
            </p>
          </Col>
        </Row>
      </Card>
    </div>
  )
}