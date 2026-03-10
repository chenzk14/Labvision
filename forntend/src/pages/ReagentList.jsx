// frontend/src/pages/ReagentList.jsx
import React, { useState, useEffect } from 'react'
import { Table, Tag, Button, Space, Modal, Descriptions, Image, message, Input } from 'antd'
import { EyeOutlined, DeleteOutlined, SearchOutlined } from '@ant-design/icons'
import { api } from '../services/api'

export default function ReagentList() {
  const [reagents, setReagents] = useState([])
  const [loading, setLoading] = useState(false)
  const [detail, setDetail] = useState(null)
  const [searchText, setSearchText] = useState('')

  const loadReagents = async () => {
    setLoading(true)
    try {
      const data = await api.listReagents()
      setReagents(data)
    } catch (e) {
      message.error('加载失败')
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => { loadReagents() }, [])

  const handleDelete = async (reagentId) => {
    Modal.confirm({
      title: `确认取出试剂 ${reagentId}？`,
      content: '此操作将标记该试剂为不在库，可随时重新录入',
      onOk: async () => {
        await api.deleteReagent(reagentId)
        message.success('已标记取出')
        loadReagents()
      }
    })
  }

  const handleViewDetail = async (reagentId) => {
    const data = await api.getReagent(reagentId)
    setDetail(data)
  }

  const filtered = reagents.filter(r =>
    r.reagent_id.includes(searchText) || r.reagent_name.includes(searchText)
  )

  const columns = [
    {
      title: '试剂ID',
      dataIndex: 'reagent_id',
      key: 'reagent_id',
      render: (id) => <strong>{id}</strong>,
    },
    { title: '名称', dataIndex: 'reagent_name', key: 'reagent_name' },
    { title: 'CAS号', dataIndex: 'cas_number', key: 'cas_number', render: v => v || '-' },
    { title: '厂商', dataIndex: 'manufacturer', key: 'manufacturer', render: v => v || '-' },
    { title: '数量', key: 'qty', render: r => r.quantity ? `${r.quantity} ${r.unit || ''}` : '-' },
    { title: '位置', dataIndex: 'location', key: 'location', render: v => v || '-' },
    {
      title: '图片数',
      dataIndex: 'image_count',
      key: 'image_count',
      render: n => <Tag color={n > 0 ? 'green' : 'red'}>{n}张</Tag>,
    },
    {
      title: '状态',
      dataIndex: 'is_active',
      key: 'is_active',
      render: v => <Tag color={v ? 'green' : 'default'}>{v ? '在库' : '已取出'}</Tag>,
    },
    {
      title: '操作',
      key: 'action',
      render: (_, record) => (
        <Space>
          <Button size="small" icon={<EyeOutlined />} onClick={() => handleViewDetail(record.reagent_id)}>
            详情
          </Button>
          <Button size="small" danger icon={<DeleteOutlined />} onClick={() => handleDelete(record.reagent_id)}>
            取出
          </Button>
        </Space>
      ),
    },
  ]

  return (
    <div>
      <div style={{ marginBottom: 16, display: 'flex', gap: 12 }}>
        <Input
          placeholder="搜索试剂ID或名称..."
          prefix={<SearchOutlined />}
          value={searchText}
          onChange={e => setSearchText(e.target.value)}
          style={{ width: 280 }}
        />
        <Button onClick={loadReagents} loading={loading}>刷新</Button>
      </div>

      <Table
        columns={columns}
        dataSource={filtered}
        rowKey="reagent_id"
        loading={loading}
        pagination={{ pageSize: 20 }}
        size="middle"
      />

      {/* 详情弹窗 */}
      <Modal
        title={`试剂详情: ${detail?.reagent?.reagent_id}`}
        open={!!detail}
        onCancel={() => setDetail(null)}
        footer={null}
        width={700}
      >
        {detail && (
          <>
            <Descriptions bordered size="small" column={2}>
              <Descriptions.Item label="试剂ID">{detail.reagent.reagent_id}</Descriptions.Item>
              <Descriptions.Item label="名称">{detail.reagent.reagent_name}</Descriptions.Item>
              <Descriptions.Item label="CAS号">{detail.reagent.cas_number || '-'}</Descriptions.Item>
              <Descriptions.Item label="厂商">{detail.reagent.manufacturer || '-'}</Descriptions.Item>
              <Descriptions.Item label="批次">{detail.reagent.batch_number || '-'}</Descriptions.Item>
              <Descriptions.Item label="有效期">{detail.reagent.expiry_date || '-'}</Descriptions.Item>
              <Descriptions.Item label="存放位置">{detail.reagent.location || '-'}</Descriptions.Item>
              <Descriptions.Item label="数量">
                {detail.reagent.quantity ? `${detail.reagent.quantity} ${detail.reagent.unit || ''}` : '-'}
              </Descriptions.Item>
              <Descriptions.Item label="备注" span={2}>{detail.reagent.notes || '-'}</Descriptions.Item>
            </Descriptions>

            <div style={{ marginTop: 16 }}>
              <strong>已注册图片 ({detail.images.length}张)</strong>
              <div style={{ display: 'flex', flexWrap: 'wrap', gap: 8, marginTop: 8 }}>
                {detail.images.map(img => (
                  <div key={img.id} style={{ textAlign: 'center' }}>
                    <Image
                      src={`http://localhost:8000${img.path}`}
                      width={100}
                      height={100}
                      style={{ objectFit: 'cover', borderRadius: 4 }}
                    />
                    <div><Tag>{img.angle}</Tag></div>
                  </div>
                ))}
              </div>
            </div>
          </>
        )}
      </Modal>
    </div>
  )
}