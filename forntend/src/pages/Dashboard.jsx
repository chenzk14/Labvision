
// frontend/src/pages/Dashboard.jsx
import React, { useState, useEffect } from 'react'
import { Card, Row, Col, Statistic, Alert, Tag, Spin } from 'antd'
import {
  ExperimentOutlined, DatabaseOutlined, ThunderboltOutlined, CheckCircleOutlined
} from '@ant-design/icons'
import { api } from '../services/api'

export default function Dashboard() {
  const [status, setStatus] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)

  useEffect(() => {
    api.getStatus()
      .then(setStatus)
      .catch(e => setError('无法连接后端服务，请确认已启动 uvicorn'))
      .finally(() => setLoading(false))
  }, [])

  if (loading) return <div style={{ textAlign: 'center', padding: 80 }}><Spin size="large" /></div>

  if (error) return (
    <Alert
      type="error"
      message="连接失败"
      description={error}
      showIcon
      style={{ maxWidth: 600 }}
    />
  )

  return (
    <div>
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
        <Alert
          type="info"
          message="快速开始"
          description="1. 前往「试剂录入」注册试剂信息和图片 → 2. 前往「实时识别」测试摄像头识别效果"
          showIcon
        />
      </Card>
    </div>
  )
}