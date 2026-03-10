// frontend/src/pages/ReagentRegister.jsx
import React, { useState, useRef, useCallback } from 'react'
import {
  Form, Input, InputNumber, Button, Upload, Steps,
  Card, Row, Col, message, Tag, Space, Divider, Alert,
  Select, DatePicker
} from 'antd'
import {
  CameraOutlined, UploadOutlined, CheckCircleOutlined,
  LoadingOutlined, PlusOutlined,
} from '@ant-design/icons'
import { api } from '../services/api'

const { Step } = Steps

export default function ReagentRegister() {
  const [currentStep, setCurrentStep] = useState(0)
  const [form] = Form.useForm()
  const [loading, setLoading] = useState(false)
  const [createdReagentId, setCreatedReagentId] = useState(null)
  const [registeredImages, setRegisteredImages] = useState([])
  const [capturedImage, setCapturedImage] = useState(null)
  const [cameraActive, setCameraActive] = useState(false)
  const videoRef = useRef(null)
  const streamRef = useRef(null)

  // 步骤1：创建试剂信息
  const handleCreateReagent = async (values) => {
    setLoading(true)
    try {
      const res = await api.createReagent(values)
      if (res.success) {
        setCreatedReagentId(res.reagent_id)
        message.success(`试剂 ${res.reagent_id} 创建成功！`)
        setCurrentStep(1)
      }
    } catch (err) {
      message.error(err.response?.data?.detail || '创建失败')
    } finally {
      setLoading(false)
    }
  }

  // 开启摄像头
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

  // 关闭摄像头
  const stopCamera = () => {
    if (streamRef.current) {
      streamRef.current.getTracks().forEach(t => t.stop())
      streamRef.current = null
      setCameraActive(false)
    }
  }

  // 拍照
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

  // 注册图片
  const registerImage = async (angle) => {
    if (!capturedImage && !uploadFile) {
      message.warning('请先拍照或上传图片')
      return
    }
    setLoading(true)
    try {
      const formData = new FormData()
      if (capturedImage) {
        formData.append('file', capturedImage.blob, 'capture.jpg')
      }
      formData.append('angle', angle)

      const res = await api.registerImage(createdReagentId, formData)
      if (res.success) {
        setRegisteredImages(prev => [...prev, {
          angle,
          url: capturedImage?.url,
          vector_id: res.vector_id,
        }])
        message.success(`✅ ${angle} 角度注册成功（向量ID: ${res.vector_id}）`)
        setCapturedImage(null)
      }
    } catch (err) {
      message.error('注册失败：' + (err.response?.data?.detail || err.message))
    } finally {
      setLoading(false)
    }
  }

  const [uploadFile, setUploadFile] = useState(null)

  const handleUpload = async ({ file }) => {
    setUploadFile(file)
    const reader = new FileReader()
    reader.onload = (e) => {
      setCapturedImage({ blob: file, url: e.target.result })
    }
    reader.readAsDataURL(file)
    return false  // 阻止自动上传
  }

  const angleConfig = [
    { key: 'front', label: '正面', color: '#1890ff', desc: '标签正面朝摄像头' },
    { key: 'side', label: '侧面', color: '#52c41a', desc: '试剂瓶侧面' },
    { key: 'top', label: '顶部', color: '#faad14', desc: '从顶部俯拍瓶盖' },
  ]

  return (
    <div>
      <Steps current={currentStep} style={{ marginBottom: 32 }}>
        <Step title="录入信息" description="填写试剂基本信息" />
        <Step title="注册图片" description="拍摄多角度图片" />
        <Step title="完成" description="试剂已录入识别系统" />
      </Steps>

      {/* 步骤1：信息录入 */}
      {currentStep === 0 && (
        <Card title="试剂基本信息">
          <Form
            form={form}
            layout="vertical"
            onFinish={handleCreateReagent}
            style={{ maxWidth: 700 }}
          >
            <Row gutter={16}>
              <Col span={12}>
                <Form.Item
                  name="reagent_name"
                  label="试剂名称"
                  rules={[{ required: true, message: '请输入试剂名称' }]}
                >
                  <Input placeholder="乙醇" size="large" />
                </Form.Item>
              </Col>
              <Col span={12}>
                <Form.Item
                  name="reagent_id"
                  label="试剂唯一ID（可选）"
                  rules={[
                    { pattern: /^[\u4e00-\u9fa5\w-]{1,20}$/, message: 'ID不超过20字符' }
                  ]}
                  extra="留空则自动生成（例：乙醇001，盐酸003）"
                >
                  <Input placeholder="留空自动生成" size="large" />
                </Form.Item>
              </Col>
            </Row>

            <Row gutter={16}>
              <Col span={8}>
                <Form.Item name="cas_number" label="CAS号">
                  <Input placeholder="64-17-5" />
                </Form.Item>
              </Col>
              <Col span={8}>
                <Form.Item name="manufacturer" label="厂商">
                  <Input placeholder="国药集团" />
                </Form.Item>
              </Col>
              <Col span={8}>
                <Form.Item name="batch_number" label="批次号">
                  <Input placeholder="2024010001" />
                </Form.Item>
              </Col>
            </Row>

            <Row gutter={16}>
              <Col span={8}>
                <Form.Item name="quantity" label="数量">
                  <InputNumber style={{ width: '100%' }} min={0} placeholder="500" />
                </Form.Item>
              </Col>
              <Col span={8}>
                <Form.Item name="unit" label="单位">
                  <Select placeholder="选择单位">
                    <Select.Option value="mL">mL</Select.Option>
                    <Select.Option value="L">L</Select.Option>
                    <Select.Option value="g">g</Select.Option>
                    <Select.Option value="kg">kg</Select.Option>
                    <Select.Option value="瓶">瓶</Select.Option>
                  </Select>
                </Form.Item>
              </Col>
              <Col span={8}>
                <Form.Item name="expiry_date" label="有效期">
                  <Input placeholder="2026-01-01" />
                </Form.Item>
              </Col>
            </Row>

            <Form.Item name="location" label="存放位置">
              <Input placeholder="A柜-3层-左侧" />
            </Form.Item>

            {/*<Form.Item name="notes" label="备注">*/}
            {/*  <Input.TextArea rows={2} placeholder="特殊注意事项..." />*/}
            {/*</Form.Item>*/}

            <Form.Item>
              <Button
                type="primary"
                htmlType="submit"
                loading={loading}
                size="large"
                style={{ width: 200 }}
              >
                下一步：注册图片
              </Button>
            </Form.Item>
          </Form>
        </Card>
      )}

      {/* 步骤2：图片注册 */}
      {currentStep === 1 && (
        <Row gutter={24}>
          <Col span={14}>
            <Card
              title={`图片采集 - 试剂: ${createdReagentId}`}
              extra={
                <Space>
                  {cameraActive ? (
                    <Button danger onClick={stopCamera}>关闭摄像头</Button>
                  ) : (
                    <Button icon={<CameraOutlined />} onClick={startCamera} type="primary">
                      开启摄像头
                    </Button>
                  )}
                </Space>
              }
            >
              {/* 摄像头预览 */}
              <div style={{
                position: 'relative',
                background: '#000',
                borderRadius: 8,
                overflow: 'hidden',
                marginBottom: 16,
                minHeight: 240,
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
                {!cameraActive && (
                  <div style={{ color: '#888', textAlign: 'center' }}>
                    <CameraOutlined style={{ fontSize: 48, marginBottom: 8 }} />
                    <div>点击"开启摄像头"或上传图片</div>
                  </div>
                )}
              </div>

              <Space style={{ marginBottom: 16, width: '100%' }} wrap>
                {cameraActive && (
                  <Button
                    size="large"
                    onClick={capturePhoto}
                    icon={<CameraOutlined />}
                    type="primary"
                    ghost
                  >
                    拍照
                  </Button>
                )}
                <Upload
                  accept="image/*"
                  showUploadList={false}
                  beforeUpload={(file) => { handleUpload({ file }); return false; }}
                >
                  <Button icon={<UploadOutlined />}>上传图片</Button>
                </Upload>
              </Space>

              {/* 已拍摄预览 */}
              {capturedImage && (
                <div style={{ marginBottom: 16 }}>
                  <img
                    src={capturedImage.url}
                    alt="preview"
                    style={{ width: '100%', maxHeight: 200, objectFit: 'contain', borderRadius: 4 }}
                  />
                  <div style={{ marginTop: 8, display: 'flex', gap: 8, flexWrap: 'wrap' }}>
                    {angleConfig.map(cfg => (
                      <Button
                        key={cfg.key}
                        style={{ borderColor: cfg.color, color: cfg.color }}
                        loading={loading}
                        onClick={() => registerImage(cfg.key)}
                      >
                        注册为{cfg.label}
                      </Button>
                    ))}
                  </div>
                </div>
              )}

              <Alert
                type="info"
                message="建议：每瓶试剂至少拍3个角度（正面、侧面、顶部），角度越多识别越准确"
                style={{ marginTop: 8 }}
              />
            </Card>
          </Col>

          <Col span={10}>
            <Card title={`已注册图片 (${registeredImages.length}张)`}>
              {registeredImages.length === 0 ? (
                <div style={{ textAlign: 'center', color: '#999', padding: '40px 0' }}>
                  尚未注册任何图片
                </div>
              ) : (
                <Row gutter={8}>
                  {registeredImages.map((img, idx) => (
                    <Col span={12} key={idx} style={{ marginBottom: 8 }}>
                      <div style={{ position: 'relative' }}>
                        <img
                          src={img.url}
                          alt={img.angle}
                          style={{ width: '100%', height: 100, objectFit: 'cover', borderRadius: 4 }}
                        />
                        <Tag
                          color={angleConfig.find(a => a.key === img.angle)?.color}
                          style={{ position: 'absolute', top: 4, left: 4, margin: 0 }}
                        >
                          {img.angle}
                        </Tag>
                        <CheckCircleOutlined
                          style={{ position: 'absolute', top: 4, right: 4, color: '#52c41a', fontSize: 16 }}
                        />
                      </div>
                    </Col>
                  ))}
                </Row>
              )}

              <Divider />

              <Button
                type="primary"
                size="large"
                style={{ width: '100%' }}
                disabled={registeredImages.length === 0}
                onClick={() => { stopCamera(); setCurrentStep(2) }}
              >
                完成注册
              </Button>
            </Card>
          </Col>
        </Row>
      )}

      {/* 步骤3：完成 */}
      {currentStep === 2 && (
        <Card style={{ textAlign: 'center', padding: '48px 0' }}>
          <CheckCircleOutlined style={{ fontSize: 64, color: '#52c41a', marginBottom: 16 }} />
          <h2>试剂 {createdReagentId} 录入完成！</h2>
          <p style={{ color: '#666', marginBottom: 24 }}>
            已注册 {registeredImages.length} 张图片到识别系统，
            摄像头现在可以自动识别此试剂
          </p>
          <Space>
            <Button
              type="primary"
              onClick={() => {
                form.resetFields()
                setCurrentStep(0)
                setCreatedReagentId(null)
                setRegisteredImages([])
                setCapturedImage(null)
              }}
            >
              继续录入下一瓶
            </Button>
          </Space>
        </Card>
      )}
    </div>
  )
}