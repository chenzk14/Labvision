// frontend/src/pages/ReagentRecognize.jsx
import React, { useState, useRef, useEffect, useCallback } from 'react'
import {
  Card, Button, Row, Col, Tag, Progress, Alert,
  Statistic, Space, Switch, Spin, Badge, List, Avatar,
  Modal, Form, Input, Select, message,
} from 'antd'
import {
  CameraOutlined, SearchOutlined, CheckCircleOutlined,
  CloseCircleOutlined, ExperimentOutlined, EditOutlined,
} from '@ant-design/icons'
import { api } from '../services/api'

export default function ReagentRecognize() {
  const [cameraActive, setCameraActive] = useState(false)
  const [autoRecognize, setAutoRecognize] = useState(false)
  const [result, setResult] = useState(null)
  const [loading, setLoading] = useState(false)
  const [history, setHistory] = useState([])
  const [showCorrectionModal, setShowCorrectionModal] = useState(false)
  const [submittingCorrection, setSubmittingCorrection] = useState(false)
  const [reagents, setReagents] = useState([])
  const [capturedImageForCorrection, setCapturedImageForCorrection] = useState(null)

  const videoRef = useRef(null)
  const streamRef = useRef(null)
  const autoTimerRef = useRef(null)
  const [form] = Form.useForm()

  const startCamera = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: 640, height: 480 }
      })
      videoRef.current.srcObject = stream
      streamRef.current = stream
      setCameraActive(true)
    } catch (err) {
      console.error(err)
    }
  }

  const stopCamera = () => {
    if (streamRef.current) {
      streamRef.current.getTracks().forEach(t => t.stop())
      streamRef.current = null
      setCameraActive(false)
      setAutoRecognize(false)
    }
  }

  const captureAndRecognize = useCallback(async () => {
    if (!videoRef.current || loading) return
    setLoading(true)

    try {
      const canvas = document.createElement('canvas')
      canvas.width = videoRef.current.videoWidth
      canvas.height = videoRef.current.videoHeight
      canvas.getContext('2d').drawImage(videoRef.current, 0, 0)

      const imageData = canvas.toDataURL('image/jpeg', 0.9)
      const b64 = imageData.split(',')[1]
      const res = await api.recognizeBase64(b64)
      setResult(res)
      setCapturedImageForCorrection(imageData)

      if (res.recognized) {
        setHistory(prev => [{
          id: Date.now(),
          reagent_id: res.reagent_id,
          reagent_name: res.reagent_name,
          confidence: res.confidence,
          timestamp: new Date().toLocaleTimeString(),
        }, ...prev.slice(0, 19)])
      }
    } catch (err) {
      console.error(err)
    } finally {
      setLoading(false)
    }
  }, [loading])

  // 自动识别
  useEffect(() => {
    if (autoRecognize && cameraActive) {
      autoTimerRef.current = setInterval(captureAndRecognize, 1500)
    } else {
      clearInterval(autoTimerRef.current)
    }
    return () => clearInterval(autoTimerRef.current)
  }, [autoRecognize, cameraActive, captureAndRecognize])

  useEffect(() => {
    loadReagents()
  }, [])

  const loadReagents = async () => {
    try {
      const data = await api.listReagents()
      setReagents(data)
    } catch (e) {
      console.error('加载试剂列表失败', e)
    }
  }

  const confidenceColor = (score) => {
    if (!score) return '#d9d9d9'
    if (score >= 0.9) return '#52c41a'
    if (score >= 0.75) return '#1890ff'
    return '#ff4d4f'
  }

  const handleOpenCorrection = () => {
    setShowCorrectionModal(true)
    form.setFieldsValue({
      original_recognition_id: result?.reagent_id,
      original_confidence: result?.confidence,
    })
  }

  const handleSubmitCorrection = async (values) => {
    setSubmittingCorrection(true)
    try {
      const b64 = capturedImageForCorrection.split(',')[1]
      const blob = await fetch(`data:image/jpeg;base64,${b64}`).then(res => res.blob())

      const formData = new FormData()
      formData.append('file', blob, 'correction.jpg')
      formData.append('corrected_reagent_id', values.corrected_reagent_id)
      formData.append('corrected_reagent_name', values.corrected_reagent_name)
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
      formData.append('correction_source', 'recognize_page')

      const res = await api.submitCorrection(formData)
      if (res.success) {
        message.success('纠错提交成功！已自动应用到识别系统')
        setShowCorrectionModal(false)
        form.resetFields()
      }
    } catch (e) {
      message.error('提交失败：' + (e.response?.data?.detail || e.message))
    } finally {
      setSubmittingCorrection(false)
    }
  }

  return (
    <Row gutter={24}>
      {/* 摄像头区域 */}
      <Col span={14}>
        <Card
          title="摄像头"
          extra={
            <Space>
              <span style={{ color: '#666' }}>自动识别</span>
              <Switch
                checked={autoRecognize}
                disabled={!cameraActive}
                onChange={setAutoRecognize}
                checkedChildren="开"
                unCheckedChildren="关"
              />
              {cameraActive ? (
                <Button danger onClick={stopCamera} size="small">关闭</Button>
              ) : (
                <Button type="primary" icon={<CameraOutlined />} onClick={startCamera} size="small">
                  开启摄像头
                </Button>
              )}
            </Space>
          }
        >
          <div style={{
            position: 'relative',
            background: '#000',
            borderRadius: 8,
            overflow: 'hidden',
            minHeight: 400,
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
          }}>
            <video
              ref={videoRef}
              autoPlay playsInline muted
              style={{ width: '100%', display: cameraActive ? 'block' : 'none' }}
            />
            {!cameraActive && (
              <div style={{ color: '#555', textAlign: 'center' }}>
                <CameraOutlined style={{ fontSize: 64, color: '#333' }} />
                <div style={{ color: '#888', marginTop: 8 }}>点击开启摄像头</div>
              </div>
            )}
            {loading && (
              <div style={{
                position: 'absolute', top: 0, left: 0, right: 0, bottom: 0,
                background: 'rgba(0,0,0,0.4)',
                display: 'flex', alignItems: 'center', justifyContent: 'center',
              }}>
                <Spin size="large" />
              </div>
            )}
          </div>

          <div style={{ marginTop: 16, textAlign: 'center' }}>
            <Button
              type="primary"
              size="large"
              icon={<SearchOutlined />}
              onClick={captureAndRecognize}
              disabled={!cameraActive || loading}
              loading={loading}
              style={{ width: 200 }}
            >
              立即识别
            </Button>
          </div>
        </Card>
      </Col>

      {/* 识别结果 */}
      <Col span={10}>
        <Card
          title="识别结果"
          style={{ marginBottom: 16 }}
          bodyStyle={{ padding: '16px 24px' }}
        >
          {!result ? (
            <div style={{ textAlign: 'center', padding: '32px 0', color: '#999' }}>
              <ExperimentOutlined style={{ fontSize: 48 }} />
              <div style={{ marginTop: 8 }}>等待识别...</div>
            </div>
          ) : (
            <>
              {/* 主结果 */}
              <div style={{
                textAlign: 'center',
                padding: '20px 0',
                borderBottom: '1px solid #f0f0f0',
                marginBottom: 16,
              }}>
                {result.recognized ? (
                  <>
                    <CheckCircleOutlined style={{ fontSize: 40, color: '#52c41a' }} />
                    <div style={{ fontSize: 24, fontWeight: 700, marginTop: 8 }}>
                      {result.reagent_name || result.reagent_id}
                    </div>
                    {result.reagent_name && result.reagent_name !== result.reagent_id && (
                      <div style={{ color: '#666' }}>ID: {result.reagent_id}</div>
                    )}
                  </>
                ) : (
                  <>
                    <CloseCircleOutlined style={{ fontSize: 40, color: '#ff4d4f' }} />
                    <div style={{ fontSize: 16, color: '#ff4d4f', marginTop: 8 }}>
                      未识别
                    </div>
                  </>
                )}
              </div>

              {/* 置信度 */}
              {result.confidence != null && (
                <div style={{ marginBottom: 16 }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 4 }}>
                    <span>置信度</span>
                    <strong style={{ color: confidenceColor(result.confidence) }}>
                      {result.confidence_pct}
                    </strong>
                  </div>
                  <Progress
                    percent={Math.round(result.confidence * 100)}
                    strokeColor={confidenceColor(result.confidence)}
                    showInfo={false}
                    size="small"
                  />
                </div>
              )}

              {/* Top-K候选 */}
              {result.candidates?.length > 0 && (
                <div>
                  <div style={{ color: '#999', fontSize: 12, marginBottom: 8 }}>候选结果</div>
                  {result.candidates.slice(0, 5).map((cand, i) => (
                    <div key={i} style={{
                      display: 'flex', justifyContent: 'space-between',
                      alignItems: 'center', padding: '4px 0',
                      borderBottom: i < 4 ? '1px solid #f5f5f5' : 'none',
                    }}>
                      <Space size={4}>
                        <Tag color={i === 0 ? 'blue' : 'default'} style={{ margin: 0 }}>
                          #{i + 1}
                        </Tag>
                        <div>
                          <span style={{ fontSize: 13 }}>{cand.reagent_name || cand.reagent_id}</span>
                          {cand.reagent_id && cand.reagent_id !== cand.reagent_name && (
                            <span style={{ color: '#666', fontSize: 12, marginLeft: 4 }}>
                              ({cand.reagent_id})
                            </span>
                          )}
                        </div>
                      </Space>
                      <span style={{
                        fontSize: 12,
                        color: confidenceColor(cand.similarity),
                        fontWeight: i === 0 ? 700 : 400,
                      }}>
                        {cand.confidence_pct}
                      </span>
                    </div>
                  ))}
                </div>
              )}

              {/* 消息 */}
              <Alert
                message={result.message}
                type={result.recognized ? 'success' : 'warning'}
                style={{ marginTop: 12 }}
                showIcon
              />

              {/* 纠错按钮 */}
              <Button
                type="primary"
                danger={!result.recognized}
                icon={<EditOutlined />}
                onClick={handleOpenCorrection}
                style={{ marginTop: 12, width: '100%' }}
              >
                {result.recognized ? '识别错误？点击纠错' : '未识别？点击纠错'}
              </Button>
            </>
          )}
        </Card>

        {/* 识别历史 */}
        <Card
          title={`识别记录 (${history.length})`}
          bodyStyle={{ padding: '8px 16px', maxHeight: 240, overflowY: 'auto' }}
          size="small"
        >
          {history.length === 0 ? (
            <div style={{ textAlign: 'center', padding: '16px 0', color: '#999', fontSize: 12 }}>
              暂无记录
            </div>
          ) : (
            history.map(item => (
              <div key={item.id} style={{
                display: 'flex', justifyContent: 'space-between',
                alignItems: 'center', padding: '6px 0',
                borderBottom: '1px solid #f5f5f5',
                fontSize: 13,
              }}>
                <Space size={8}>
                  <Badge color={confidenceColor(item.confidence)} />
                  <div>
                    <strong>{item.reagent_name || item.reagent_id}</strong>
                    {item.reagent_id && item.reagent_id !== item.reagent_name && (
                      <span style={{ color: '#666', marginLeft: 4 }}>
                        ({item.reagent_id})
                      </span>
                    )}
                  </div>
                </Space>
                <Space size={12}>
                  <span style={{ color: confidenceColor(item.confidence) }}>
                    {(item.confidence * 100).toFixed(1)}%
                  </span>
                  <span style={{ color: '#bbb', fontSize: 11 }}>{item.timestamp}</span>
                </Space>
              </div>
            ))
          )}
        </Card>
      </Col>

      {/* 纠错弹窗 */}
      <Modal
        title="纠错识别结果"
        open={showCorrectionModal}
        onCancel={() => {
          setShowCorrectionModal(false)
          form.resetFields()
        }}
        footer={null}
        width={500}
      >
        <Alert
          message="纠错说明"
          description="请选择正确的试剂ID，系统将自动将此图片注册到识别引擎中，提升后续识别准确率。"
          type="info"
          showIcon
          style={{ marginBottom: 16 }}
        />
        <Form
          form={form}
          layout="vertical"
          onFinish={handleSubmitCorrection}
        >
          {capturedImageForCorrection && (
            <div style={{ marginBottom: 16, textAlign: 'center' }}>
              <img
                src={capturedImageForCorrection}
                alt="captured"
                style={{ maxWidth: '100%', maxHeight: 200, objectFit: 'contain', borderRadius: 4 }}
              />
            </div>
          )}
          <Form.Item
            name="corrected_reagent_id"
            label="正确的试剂ID"
            rules={[{ required: true, message: '请选择试剂ID' }]}
          >
            <Select
              showSearch
              placeholder="选择正确的试剂"
              filterOption={(input, option) =>
                (option?.label ?? '').toLowerCase().includes(input.toLowerCase())
              }
              options={reagents.map(r => ({ label: `${r.reagent_id} - ${r.reagent_name}`, value: r.reagent_id }))}
            />
          </Form.Item>
          <Form.Item
            name="corrected_reagent_name"
            label="试剂名称"
            rules={[{ required: true, message: '请输入试剂名称' }]}
          >
            <Input placeholder="如：乙醇" />
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
            <Select
              options={[
                { label: '立即应用到识别系统', value: true },
                { label: '暂不应用', value: false },
              ]}
            />
          </Form.Item>
          <Form.Item>
            <Button
              type="primary"
              htmlType="submit"
              loading={submittingCorrection}
              block
            >
              提交纠错
            </Button>
          </Form.Item>
        </Form>
      </Modal>
    </Row>
  )
}