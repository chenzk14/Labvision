// frontend/src/pages/ReagentList.jsx
import React, { useState, useEffect, useRef, useCallback } from 'react'
import { Table, Tag, Button, Space, Modal, Descriptions, Image, message, Input, Popconfirm, Upload, Alert, Row, Col } from 'antd'
import { EyeOutlined, DeleteOutlined, SearchOutlined, ExclamationCircleOutlined, CameraOutlined, UploadOutlined, PlusOutlined } from '@ant-design/icons'
import { api } from '../services/api'

export default function ReagentList() {
  const [reagents, setReagents] = useState([])
  const [loading, setLoading] = useState(false)
  const [detail, setDetail] = useState(null)
  const [searchText, setSearchText] = useState('')
  const [addImageModal, setAddImageModal] = useState(false)
  const [cameraActive, setCameraActive] = useState(false)
  const [capturedImage, setCapturedImage] = useState(null)
  const [uploadFile, setUploadFile] = useState(null)
  const [registering, setRegistering] = useState(false)
  const videoRef = useRef(null)
  const streamRef = useRef(null)

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

  // 页面获得焦点时自动刷新
  useEffect(() => {
    const handleVisibilityChange = () => {
      if (!document.hidden) {
        loadReagents()
      }
    }
    document.addEventListener('visibilitychange', handleVisibilityChange)
    return () => document.removeEventListener('visibilitychange', handleVisibilityChange)
  }, [])

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

  const handlePermanentDelete = async (reagentId) => {
    Modal.confirm({
      title: `永久删除试剂 ${reagentId}？`,
      icon: <ExclamationCircleOutlined style={{ color: '#ff4d4f' }} />,
      content: (
        <div>
          <p style={{ color: '#ff4d4f', fontWeight: 600 }}>此操作不可恢复！</p>
          <p>将删除以下内容：</p>
          <ul style={{ marginLeft: 20 }}>
            <li>数据库中的试剂记录</li>
            <li>FAISS 索引中的所有特征向量</li>
            <li>所有图片文件</li>
          </ul>
        </div>
      ),
      okText: '确认删除',
      okType: 'danger',
      cancelText: '取消',
      onOk: async () => {
        try {
          const res = await api.deleteReagentPermanent(reagentId)
          message.success(
            `删除成功：${res.deleted_vectors} 个特征，${res.deleted_files} 个文件`
          )
          loadReagents()
        } catch (err) {
          message.error('删除失败：' + (err.response?.data?.detail || err.message))
        }
      }
    })
  }

  const handleViewDetail = async (reagentId) => {
    const data = await api.getReagent(reagentId)
    setDetail(data)
  }

  const handleDeleteImage = async (imgId) => {
    if (!detail?.reagent?.reagent_id) return
    try {
      await api.deleteReagentImage(detail.reagent.reagent_id, imgId)
      message.success('图片已删除，并已更新检索索引')
      await handleViewDetail(detail.reagent.reagent_id)
      loadReagents()
    } catch (e) {
      message.error('删除失败：' + (e.response?.data?.detail || e.message))
    }
  }

  const handleOpenAddImage = () => {
    setAddImageModal(true)
    setCapturedImage(null)
    setUploadFile(null)
    setCameraActive(false)
  }

  const handleCloseAddImage = () => {
    setAddImageModal(false)
    stopCamera()
    setCapturedImage(null)
    setUploadFile(null)
  }

  const startCamera = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: 640, height: 480, facingMode: 'environment' }
      })
      if (videoRef.current) {
        videoRef.current.srcObject = stream
        streamRef.current = stream
        setCameraActive(true)
      }
    } catch (err) {
      message.error('无法访问摄像头：' + err.message)
    }
  }

  const stopCamera = () => {
    if (streamRef.current) {
      streamRef.current.getTracks().forEach(t => t.stop())
      streamRef.current = null
      setCameraActive(false)
    }
  }

  const capturePhoto = useCallback(() => {
    if (!videoRef.current) return
    const canvas = document.createElement('canvas')
    canvas.width = videoRef.current.videoWidth
    canvas.height = videoRef.current.videoHeight
    const ctx = canvas.getContext('2d')
    ctx.drawImage(videoRef.current, 0, 0)
    canvas.toBlob(blob => {
      setCapturedImage({ blob, url: canvas.toDataURL('image/jpeg') })
    }, 'image/jpeg', 0.92)
  }, [])

  const handleUpload = async ({ file }) => {
    setUploadFile(file)
    const reader = new FileReader()
    reader.onload = (e) => {
      setCapturedImage({ blob: file, url: e.target.result })
    }
    reader.readAsDataURL(file)
    return false
  }

  const registerImage = async (angle) => {
    if (!capturedImage) {
      message.warning('请先拍照或上传图片')
      return
    }
    setRegistering(true)
    try {
      const formData = new FormData()
      formData.append('file', capturedImage.blob, 'capture.jpg')
      formData.append('angle', angle)

      const res = await api.registerImage(detail.reagent.reagent_id, formData)
      if (res.success) {
        message.success(`✅ ${angle} 角度注册成功`)
        setCapturedImage(null)
        setUploadFile(null)
        handleViewDetail(detail.reagent.reagent_id)
      }
    } catch (err) {
      message.error('注册失败：' + (err.response?.data?.detail || err.message))
    } finally {
      setRegistering(false)
    }
  }

  const angleConfig = [
    { key: 'front', label: '正面', color: '#1890ff', desc: '标签正面朝摄像头' },
    { key: 'side', label: '侧面', color: '#52c41a', desc: '试剂瓶侧面' },
    { key: 'top', label: '顶部', color: '#faad14', desc: '从顶部俯拍瓶盖' },
  ]

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
          {/*<Button */}
          {/*  size="small" */}
          {/*  danger */}
          {/*  icon={<DeleteOutlined />} */}
          {/*  onClick={() => handleDelete(record.reagent_id)}*/}
          {/*>*/}
          {/*  取出*/}
          {/*</Button>*/}
          <Popconfirm
            title="永久删除"
            description="此操作将永久删除试剂的所有数据，不可恢复！"
            onConfirm={() => handlePermanentDelete(record.reagent_id)}
            okText="确认删除"
            cancelText="取消"
            okButtonProps={{ danger: true }}
          >
            <Button 
              size="small" 
              type="primary" 
              danger
              icon={<DeleteOutlined />}
            >
              永久删除
            </Button>
          </Popconfirm>
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
        <Button 
          onClick={loadReagents} 
          loading={loading}
          type="primary"
        >
          刷新列表
        </Button>
        <Button 
          onClick={async () => {
            try {
              const res = await api.syncImageCounts()
              message.success(res.message)
              if (res.details && res.details.length > 0) {
                console.log('同步详情:', res.details)
              }
              loadReagents()
            } catch (e) {
              message.error('同步失败：' + (e.response?.data?.detail || e.message))
            }
          }}
        >
          同步图片数
        </Button>
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
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 8 }}>
                <strong>已注册图片 ({detail.images.length}张)</strong>
                <Button 
                  type="primary" 
                  size="small" 
                  icon={<PlusOutlined />}
                  onClick={handleOpenAddImage}
                >
                  添加图片
                </Button>
              </div>
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
                    <div style={{ marginTop: 4 }}>
                      <Popconfirm
                        title="确认删除这张图片？"
                        description="删除后会自动更新检索索引，识别结果会立即生效。"
                        okText="删除"
                        okType="danger"
                        cancelText="取消"
                        onConfirm={() => handleDeleteImage(img.id)}
                      >
                        <Button danger size="small" icon={<DeleteOutlined />}>删除</Button>
                      </Popconfirm>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </>
        )}
      </Modal>

      {/* 添加图片弹窗 */}
      <Modal
        title={`添加图片 - ${detail?.reagent?.reagent_id}`}
        open={addImageModal}
        onCancel={handleCloseAddImage}
        footer={null}
        width={800}
      >
        <Row gutter={24}>
          <Col span={16}>
            <div style={{
              position: 'relative',
              background: '#000',
              borderRadius: 8,
              overflow: 'hidden',
              marginBottom: 16,
              minHeight: 320,
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
            }}>
              <video
                ref={videoRef}
                autoPlay
                playsInline
                muted
                style={{
                  width: '100%',
                  display: cameraActive ? 'block' : 'none'
                }}
              />
              {!cameraActive && !capturedImage && (
                <div style={{ color: '#888', textAlign: 'center' }}>
                  <CameraOutlined style={{ fontSize: 48, marginBottom: 8 }} />
                  <div>点击"开启摄像头"或上传图片</div>
                </div>
              )}
              {capturedImage && (
                <img
                  src={capturedImage.url}
                  alt="preview"
                  style={{ width: '100%', maxHeight: 320, objectFit: 'contain' }}
                />
              )}
            </div>

            <Space style={{ marginBottom: 16, width: '100%' }} wrap>
              {cameraActive ? (
                <Space>
                  <Button
                    size="large"
                    onClick={capturePhoto}
                    icon={<CameraOutlined />}
                    type="primary"
                  >
                    拍照
                  </Button>
                  <Button danger onClick={stopCamera}>关闭摄像头</Button>
                </Space>
              ) : (
                <Button 
                  icon={<CameraOutlined />} 
                  onClick={startCamera} 
                  type="primary"
                  size="large"
                >
                  开启摄像头
                </Button>
              )}
              <Upload
                accept="image/*"
                showUploadList={false}
                beforeUpload={handleUpload}
              >
                <Button icon={<UploadOutlined />} size="large">上传图片</Button>
              </Upload>
            </Space>

            {capturedImage && (
              <div>
                <div style={{ marginBottom: 8, fontWeight: 500 }}>选择角度注册：</div>
                <Space wrap>
                  {angleConfig.map(cfg => (
                    <Button
                      key={cfg.key}
                      style={{ borderColor: cfg.color, color: cfg.color }}
                      loading={registering}
                      onClick={() => registerImage(cfg.key)}
                    >
                      注册为{cfg.label}
                    </Button>
                  ))}
                </Space>
              </div>
            )}

            <Alert
              type="info"
              message="建议：添加不同角度的图片可以提高识别准确率"
              style={{ marginTop: 16 }}
            />
          </Col>

          <Col span={8}>
            <div style={{ background: '#f5f5f5', padding: 16, borderRadius: 8 }}>
              <h4>已注册图片 ({detail?.images?.length || 0}张)</h4>
              <div style={{ display: 'flex', flexWrap: 'wrap', gap: 8 }}>
                {detail?.images?.map(img => (
                  <div key={img.id} style={{ textAlign: 'center' }}>
                    <Image
                      src={`http://localhost:8000${img.path}`}
                      width={80}
                      height={80}
                      style={{ objectFit: 'cover', borderRadius: 4 }}
                    />
                    <div><Tag style={{ fontSize: 10 }}>{img.angle}</Tag></div>
                  </div>
                ))}
              </div>
            </div>
          </Col>
        </Row>
      </Modal>
    </div>
  )
}