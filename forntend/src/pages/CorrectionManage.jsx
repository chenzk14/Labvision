// frontend/src/pages/CorrectionManage.jsx
import React, { useState, useEffect, useRef } from 'react'
import {
  Card, Button, Row, Col, Tag, Table, Space, Modal, Form,
  Input, Select, Upload, message, Alert, Statistic, Popconfirm,
  Descriptions, Image, Switch, Drawer, Divider, Spin,
} from 'antd'
import {
  CameraOutlined, UploadOutlined, CheckCircleOutlined,
  CloseCircleOutlined, ExperimentOutlined, ReloadOutlined,
  DeleteOutlined, CheckOutlined, ExclamationCircleOutlined,
  WarningOutlined, InfoCircleOutlined, FileImageOutlined,
} from '@ant-design/icons'
import { api } from '../services/api'
import ImageCropper from '../components/ImageCropper'

export default function CorrectionManage() {
  const [corrections, setCorrections] = useState([])
  const [stats, setStats] = useState(null)
  const [loading, setLoading] = useState(false)
  const [submitting, setSubmitting] = useState(false)
  const [showSubmitModal, setShowSubmitModal] = useState(false)
  const [showCameraModal, setShowCameraModal] = useState(false)
  const [selectedCorrection, setSelectedCorrection] = useState(null)
  const [cameraActive, setCameraActive] = useState(false)
  const [capturedImage, setCapturedImage] = useState(null)
  const [reagents, setReagents] = useState([])
  const [cropPixels, setCropPixels] = useState(null)

  const videoRef = useRef(null)
  const streamRef = useRef(null)
  const [form] = Form.useForm()

  const loadCorrections = async () => {
    setLoading(true)
    try {
      const data = await api.getCorrections({ limit: 100 })
      setCorrections(data)
    } catch (e) {
      message.error('加载纠错记录失败')
    } finally {
      setLoading(false)
    }
  }

  const loadStats = async () => {
    try {
      const data = await api.getCorrectionStatistics()
      setStats(data)
    } catch (e) {
      console.error('加载统计失败', e)
    }
  }

  const loadReagents = async () => {
    try {
      const data = await api.listReagents()
      setReagents(data)
    } catch (e) {
      console.error('加载试剂列表失败', e)
    }
  }

  useEffect(() => {
    loadCorrections()
    loadStats()
    loadReagents()
  }, [])

  const startCamera = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: 640, height: 480 }
      })
      videoRef.current.srcObject = stream
      streamRef.current = stream
      setCameraActive(true)
    } catch (err) {
      message.error('无法开启摄像头：' + err.message)
    }
  }

  const stopCamera = () => {
    if (streamRef.current) {
      streamRef.current.getTracks().forEach(t => t.stop())
      streamRef.current = null
      setCameraActive(false)
    }
  }

  const capturePhoto = () => {
    if (!videoRef.current) return
    const canvas = document.createElement('canvas')
    canvas.width = videoRef.current.videoWidth
    canvas.height = videoRef.current.videoHeight
    const ctx = canvas.getContext('2d')
    ctx.drawImage(videoRef.current, 0, 0)
    const b64 = canvas.toDataURL('image/jpeg', 0.9)

    canvas.toBlob(blob => {
      setCapturedImage({ blob, url: b64 })
      setCropPixels(null)
    }, 'image/jpeg', 0.92)
  }

  const handleUpload = ({ file }) => {
    const reader = new FileReader()
    reader.onload = (e) => {
      setCapturedImage({ blob: file, url: e.target.result })
      setCropPixels(null)
    }
    reader.readAsDataURL(file)
    return false
  }

  const handleSubmitCorrection = async (values) => {
    if (!capturedImage) {
      message.warning('请先拍照或上传图片')
      return
    }

    setSubmitting(true)
    try {
      const formData = new FormData()
      formData.append('file', capturedImage.blob, 'correction.jpg')
      formData.append('corrected_reagent_id', values.corrected_reagent_id)
      formData.append('corrected_reagent_name', values.corrected_reagent_name)
      if (cropPixels) {
        formData.append('crop_x1', cropPixels.x1)
        formData.append('crop_y1', cropPixels.y1)
        formData.append('crop_x2', cropPixels.x2)
        formData.append('crop_y2', cropPixels.y2)
      }
      if (values.original_recognition_id) {
        formData.append('original_recognition_id', values.original_recognition_id)
      }
      if (values.original_confidence) {
        formData.append('original_confidence', values.original_confidence)
      }
      if (values.notes) {
        formData.append('notes', values.notes)
      }
      formData.append('apply_immediately', values.apply_immediately)
      formData.append('correction_source', 'web')

      const res = await api.submitCorrection(formData)
      if (res.success) {
        message.success('纠错提交成功！')
        setShowSubmitModal(false)
        setShowCameraModal(false)
        setCapturedImage(null)
        setCropPixels(null)
        form.resetFields()
        loadCorrections()
        loadStats()
      }
    } catch (e) {
      message.error('提交失败：' + (e.response?.data?.detail || e.message))
    } finally {
      setSubmitting(false)
    }
  }

  const handleApplyCorrection = async (id) => {
    try {
      const res = await api.applyCorrection(id)
      if (res.success) {
        message.success('纠错已应用到识别系统')
        loadCorrections()
        loadStats()
      }
    } catch (e) {
      message.error('应用失败：' + (e.response?.data?.detail || e.message))
    }
  }

  const handleBatchApply = async () => {
    const unapplied = corrections.filter(c => !c.is_applied)
    if (unapplied.length === 0) {
      message.info('没有未应用的纠错')
      return
    }

    Modal.confirm({
      title: '批量应用纠错',
      content: `确定要应用 ${unapplied.length} 个未应用的纠错吗？`,
      onOk: async () => {
        try {
          const ids = unapplied.map(c => c.id)
          const res = await api.batchApplyCorrections(ids)
          message.success(`成功应用 ${res.success_count} 个纠错`)
          loadCorrections()
          loadStats()
        } catch (e) {
          message.error('批量应用失败')
        }
      }
    })
  }

  const handleDeleteCorrection = async (id) => {
    try {
      await api.deleteCorrection(id)
      message.success('纠错记录已删除')
      loadCorrections()
      loadStats()
    } catch (e) {
      message.error('删除失败：' + (e.response?.data?.detail || e.message))
    }
  }

  const columns = [
    {
      title: 'ID',
      dataIndex: 'id',
      key: 'id',
      width: 60,
    },
    {
      title: '时间',
      dataIndex: 'timestamp',
      key: 'timestamp',
      width: 160,
      render: (ts) => ts ? new Date(ts).toLocaleString('zh-CN') : '-',
    },
    {
      title: '原识别',
      key: 'original',
      width: 150,
      render: (_, record) => (
        <div>
          <div style={{ fontWeight: 600 }}>{record.original_recognition_id || '未识别'}</div>
          {record.original_confidence && (
            <Tag color="orange">{(record.original_confidence * 100).toFixed(1)}%</Tag>
          )}
        </div>
      ),
    },
    {
      title: '纠正为',
      key: 'corrected',
      width: 150,
      render: (_, record) => (
        <div>
          <div style={{ fontWeight: 600, color: '#52c41a' }}>{record.corrected_reagent_id}</div>
          <div style={{ fontSize: 12, color: '#666' }}>{record.corrected_reagent_name}</div>
        </div>
      ),
    },
    {
      title: '状态',
      key: 'status',
      width: 120,
      render: (_, record) => (
        <Space direction="vertical" size={4}>
          <Tag color={record.is_applied ? 'green' : 'orange'} icon={record.is_applied ? <CheckOutlined /> : <WarningOutlined />}>
            {record.is_applied ? '已应用' : '未应用'}
          </Tag>
          {record.is_exported && <Tag color="blue">已导出</Tag>}
        </Space>
      ),
    },
    {
      title: '来源',
      dataIndex: 'correction_source',
      key: 'correction_source',
      width: 80,
      render: (source) => source ? <Tag>{source}</Tag> : '-',
    },
    {
      title: '操作',
      key: 'action',
      width: 180,
      render: (_, record) => (
        <Space size="small">
          {!record.is_applied && (
            <Button
              size="small"
              type="primary"
              icon={<CheckOutlined />}
              onClick={() => handleApplyCorrection(record.id)}
            >
              应用
            </Button>
          )}
          <Popconfirm
            title="删除纠错记录"
            description="确定要删除这条纠错记录吗？"
            onConfirm={() => handleDeleteCorrection(record.id)}
            okText="确定"
            cancelText="取消"
          >
            <Button size="small" danger icon={<DeleteOutlined />}>
              删除
            </Button>
          </Popconfirm>
        </Space>
      ),
    },
  ]

  return (
    <div>
      <Row gutter={16} style={{ marginBottom: 24 }}>
        <Col span={6}>
          <Card>
            <Statistic
              title="总纠错数"
              value={stats?.correction_count || 0}
              prefix={<ExclamationCircleOutlined />}
              valueStyle={{ color: '#1890ff' }}
            />
          </Card>
        </Col>
        <Col span={6}>
          <Card>
            <Statistic
              title="已应用"
              value={stats?.applied_count || 0}
              prefix={<CheckCircleOutlined />}
              valueStyle={{ color: '#52c41a' }}
            />
          </Card>
        </Col>
        <Col span={6}>
          <Card>
            <Statistic
              title="纠错比例"
              value={stats?.correction_ratio || '0%'}
              prefix={<WarningOutlined />}
              valueStyle={{ color: '#faad14' }}
            />
          </Card>
        </Col>
        <Col span={6}>
          <Card>
            <Statistic
              title="涉及试剂"
              value={stats?.unique_corrected_reagents || 0}
              prefix={<ExperimentOutlined />}
              valueStyle={{ color: '#722ed1' }}
            />
          </Card>
        </Col>
      </Row>

      <Card
        title="纠错管理"
        extra={
          <Space>
            <Button
              icon={<ReloadOutlined />}
              onClick={() => {
                loadCorrections()
                loadStats()
              }}
              loading={loading}
            >
              刷新
            </Button>
            <Button
              type="primary"
              icon={<CameraOutlined />}
              onClick={() => setShowCameraModal(true)}
            >
              摄像头纠错
            </Button>
            {/*<Button*/}
            {/*  icon={<UploadOutlined />}*/}
            {/*  onClick={() => setShowSubmitModal(true)}*/}
            {/*>*/}
            {/*  上传纠错*/}
            {/*</Button>*/}
            <Button
              icon={<CheckOutlined />}
              onClick={handleBatchApply}
            >
              批量应用
            </Button>
          </Space>
        }
      >
        {/*<Alert*/}
        {/*  message="纠错系统说明"*/}
        {/*  description={*/}
        {/*    <div>*/}
        {/*      <p>当识别系统错误识别试剂时，可以通过纠错功能提交正确的样本：</p>*/}
        {/*      <ul>*/}
        {/*        <li><strong>摄像头纠错：</strong>实时拍摄并提交纠错</li>*/}
        {/*        <li><strong>上传纠错：</strong>上传已有的正确图片</li>*/}
        {/*        <li><strong>应用纠错：</strong>将纠错样本应用到识别系统，提升识别准确率</li>*/}
        {/*        <li><strong>批量应用：</strong>一次性应用所有未应用的纠错</li>*/}
        {/*      </ul>*/}
        {/*    </div>*/}
        {/*  }*/}
        {/*  type="info"*/}
        {/*  showIcon*/}
        {/*  style={{ marginBottom: 16 }}*/}
        {/*/>*/}

        <Table
          columns={columns}
          dataSource={corrections}
          rowKey="id"
          loading={loading}
          pagination={{ pageSize: 20 }}
          size="middle"
          scroll={{ x: 1000 }}
        />
      </Card>

      {/* 摄像头纠错弹窗 */}
      <Modal
        title="摄像头纠错"
        open={showCameraModal}
        onCancel={() => {
          setShowCameraModal(false)
          stopCamera()
          setCapturedImage(null)
          form.resetFields()
        }}
        footer={null}
        width={800}
      >
        <Row gutter={24}>
          <Col span={14}>
            <Card size="small" title="摄像头">
              <div style={{
                position: 'relative',
                background: '#000',
                borderRadius: 8,
                overflow: 'hidden',
                minHeight: 300,
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
              }}>
                <video
                  ref={videoRef}
                  autoPlay playsInline muted
                  style={{ width: '100%', display: cameraActive ? 'block' : 'none' }}
                />
                {!cameraActive && !capturedImage && (
                  <div style={{ color: '#888', textAlign: 'center' }}>
                    <CameraOutlined style={{ fontSize: 48, marginBottom: 8 }} />
                    <div>点击开启摄像头</div>
                  </div>
                )}
                {capturedImage && (
                  <ImageCropper
                    src={capturedImage.url}
                    height={260}
                    onChange={(c) => setCropPixels(c)}
                  />
                )}
              </div>
              <Space style={{ marginTop: 12, width: '100%' }} wrap>
                {cameraActive && !capturedImage && (
                  <Button
                    type="primary"
                    icon={<CameraOutlined />}
                    onClick={capturePhoto}
                  >
                    拍照
                  </Button>
                )}
                {cameraActive && (
                  <Button danger onClick={stopCamera}>关闭摄像头</Button>
                )}
                {!cameraActive && !capturedImage && (
                  <Button type="primary" icon={<CameraOutlined />} onClick={startCamera}>
                    开启摄像头
                  </Button>
                )}
                {capturedImage && (
                  <Button onClick={() => setCapturedImage(null)}>重新拍摄</Button>
                )}
              </Space>
            </Card>
          </Col>
          <Col span={10}>
            <Card size="small" title="纠错信息">
              <Form
                form={form}
                layout="vertical"
                onFinish={handleSubmitCorrection}
              >
                <Form.Item
                  name="corrected_reagent_id"
                  label="正确的试剂名称"
                  rules={[{ required: true, message: '请选择试剂' }]}
                >
                  <Select
                    showSearch
                    placeholder="按名称搜索/选择"
                    filterOption={(input, option) =>
                      (option?.label ?? '').toLowerCase().includes(input.toLowerCase())
                    }
                    options={reagents.map(r => ({ label: r.reagent_name, value: r.reagent_id }))}
                    onChange={(rid) => {
                      const r = reagents.find(x => x.reagent_id === rid)
                      form.setFieldsValue({ corrected_reagent_name: r?.reagent_name || '' })
                    }}
                  />
                </Form.Item>
                <Form.Item
                  name="corrected_reagent_name"
                  hidden
                >
                  <Input />
                </Form.Item>
                <Form.Item
                  name="original_recognition_id"
                  label="原识别结果（可选）"
                >
                  <Input placeholder="如：乙醇002" />
                </Form.Item>
                {/* <Form.Item
                  name="original_confidence"
                  label="原识别置信度（可选）"
                >
                  <Input type="number" step="0.01" min="0" max="1" placeholder="0.75" />
                </Form.Item> */}
                <Form.Item
                  name="notes"
                  label="备注（可选）"
                >
                  <Input.TextArea rows={2} placeholder="如：标签模糊导致误识别" />
                </Form.Item>
                <Form.Item
                  name="apply_immediately"
                  valuePropName="checked"
                  initialValue={true}
                >
                  <Switch checkedChildren="立即应用" unCheckedChildren="暂不应用" />
                </Form.Item>
                <Form.Item>
                  <Button
                    type="primary"
                    htmlType="submit"
                    loading={submitting}
                    block
                    disabled={!capturedImage}
                  >
                    提交纠错
                  </Button>
                </Form.Item>
              </Form>
            </Card>
          </Col>
        </Row>
      </Modal>

      {/* 上传纠错弹窗 */}
      <Modal
        title="上传纠错"
        open={showSubmitModal}
        onCancel={() => {
          setShowSubmitModal(false)
          setCapturedImage(null)
          form.resetFields()
        }}
        footer={null}
        width={600}
      >
        <Form
          form={form}
          layout="vertical"
          onFinish={handleSubmitCorrection}
        >
          <Form.Item label="上传正确图片">
            <Upload
              accept="image/*"
              showUploadList={false}
              beforeUpload={handleUpload}
            >
              <Button icon={<UploadOutlined />}>选择图片</Button>
            </Upload>
            {capturedImage && (
              <div style={{ marginTop: 12 }}>
                <ImageCropper
                  src={capturedImage.url}
                  height={240}
                  onChange={(c) => setCropPixels(c)}
                />
              </div>
            )}
          </Form.Item>
          <Form.Item
            name="corrected_reagent_id"
            label="正确的试剂名称"
            rules={[{ required: true, message: '请选择试剂' }]}
          >
            <Select
              showSearch
              placeholder="按名称搜索/选择"
              filterOption={(input, option) =>
                (option?.label ?? '').toLowerCase().includes(input.toLowerCase())
              }
              options={reagents.map(r => ({ label: r.reagent_name, value: r.reagent_id }))}
              onChange={(rid) => {
                const r = reagents.find(x => x.reagent_id === rid)
                form.setFieldsValue({ corrected_reagent_name: r?.reagent_name || '' })
              }}
            />
          </Form.Item>
          <Form.Item
            name="corrected_reagent_name"
            hidden
          >
            <Input />
          </Form.Item>
          <Form.Item
            name="original_recognition_id"
            label="原识别结果（可选）"
          >
            <Input placeholder="如：乙醇002" />
          </Form.Item>
          {/* <Form.Item
            name="original_confidence"
            label="原识别置信度（可选）"
          >
            <Input type="number" step="0.01" min="0" max="1" placeholder="0.75" />
          </Form.Item> */}
          <Form.Item
            name="notes"
            label="备注（可选）"
          >
            <Input.TextArea rows={2} placeholder="如：标签模糊导致误识别" />
          </Form.Item>
          <Form.Item
            name="apply_immediately"
            valuePropName="checked"
            initialValue={true}
          >
            <Switch checkedChildren="立即应用" unCheckedChildren="暂不应用" />
          </Form.Item>
          <Form.Item>
            <Button
              type="primary"
              htmlType="submit"
              loading={submitting}
              block
              disabled={!capturedImage}
            >
              提交纠错
            </Button>
          </Form.Item>
        </Form>
      </Modal>
    </div>
  )
}