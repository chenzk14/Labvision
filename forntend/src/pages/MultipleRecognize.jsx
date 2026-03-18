// frontend/src/pages/MultipleRecognize.jsx
import React, { useState, useRef, useCallback, useEffect } from 'react'
import {
  Card, Button, Row, Col, Tag, Progress, Alert,
  Statistic, Space, Switch, Spin, List, Avatar,
  Slider, Upload, message, Drawer, Divider,
  Modal, Form, Input, Select,
} from 'antd'
import {
  CameraOutlined, SearchOutlined, CheckCircleOutlined,
  CloseCircleOutlined, ExperimentOutlined, UploadOutlined,
  AppstoreOutlined, PictureOutlined, SettingOutlined,
  EditOutlined,
} from '@ant-design/icons'
import { api } from '../services/api'
import ImageCropper from '../components/ImageCropper'

export default function MultipleRecognize() {
  const [cameraActive, setCameraActive] = useState(false)
  const [autoRecognize, setAutoRecognize] = useState(false)
  const [result, setResult] = useState(null)
  const [loading, setLoading] = useState(false)
  const [history, setHistory] = useState([])
  const [minConfidence, setMinConfidence] = useState(0.5)
  const [showSettings, setShowSettings] = useState(false)
  const [capturedImage, setCapturedImage] = useState(null)
  const [selectedObject, setSelectedObject] = useState(null)
  const [showCorrectionModal, setShowCorrectionModal] = useState(false)
  const [submittingCorrection, setSubmittingCorrection] = useState(false)
  const [reagents, setReagents] = useState([])
  const [correctionObject, setCorrectionObject] = useState(null)
  const [cropPixels, setCropPixels] = useState(null)
  const [isNewReagentMode, setIsNewReagentMode] = useState(false)
  const [creatingReagent, setCreatingReagent] = useState(false)

  const videoRef = useRef(null)
  const canvasRef = useRef(null)
  const streamRef = useRef(null)
  const autoTimerRef = useRef(null)
  const [form] = Form.useForm()

  const startCamera = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: 1280, height: 720 }
      })
      videoRef.current.srcObject = stream
      streamRef.current = stream
      setCameraActive(true)
    } catch (err) {
      console.error(err)
      message.error('无法开启摄像头')
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
      setCapturedImage(imageData)

      const b64 = imageData.split(',')[1]
      const res = await api.recognizeMultipleBase64(b64, minConfidence, 5)
      setResult(res)

      if (res.recognized_objects?.length > 0) {
        setHistory(prev => [{
          id: Date.now(),
          total: res.total_objects,
          recognized: res.recognized_count,
          unrecognized: res.unrecognized_count,
          timestamp: new Date().toLocaleTimeString(),
          image: imageData,
          result: res,
        }, ...prev.slice(0, 9)])
      }
    } catch (err) {
      console.error(err)
      message.error('识别失败')
    } finally {
      setLoading(false)
    }
  }, [loading, minConfidence])

  useEffect(() => {
    if (autoRecognize && cameraActive) {
      autoTimerRef.current = setInterval(captureAndRecognize, 2000)
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

  const handleUpload = async (file) => {
    setLoading(true)
    try {
      const res = await api.recognizeMultipleFile(file, minConfidence, 5)
      setResult(res)

      const reader = new FileReader()
      reader.onload = (e) => {
        setCapturedImage(e.target.result)
        if (res.recognized_objects?.length > 0) {
          setHistory(prev => [{
            id: Date.now(),
            total: res.total_objects,
            recognized: res.recognized_count,
            unrecognized: res.unrecognized_count,
            timestamp: new Date().toLocaleTimeString(),
            image: e.target.result,
            result: res,
          }, ...prev.slice(0, 9)])
        }
      }
      reader.readAsDataURL(file)
    } catch (err) {
      console.error(err)
      message.error('识别失败')
    } finally {
      setLoading(false)
    }
    return false
  }

  const drawBoundingBoxes = () => {
    if (!capturedImage || !result) return null

    const img = new Image()
    img.onload = () => {
      const canvas = canvasRef.current
      if (!canvas) return null

      const ctx = canvas.getContext('2d')
      canvas.width = img.naturalWidth
      canvas.height = img.naturalHeight

      ctx.drawImage(img, 0, 0)

      const labelHeight = 24
      const labelPadding = 4
      const allObjects = [
        ...result.recognized_objects.map(obj => ({ ...obj, type: 'recognized' })),
        ...result.unrecognized_objects.map(obj => ({ ...obj, type: 'unrecognized' }))
      ]

      const labelPositions = []

      allObjects.forEach((obj) => {
        const [x1, y1, x2, y2] = obj.bbox
        const color = obj.type === 'recognized' ? '#52c41a' : '#ff4d4f'
        const labelWidth = 200

        ctx.strokeStyle = color
        ctx.lineWidth = 3
        ctx.strokeRect(x1, y1, x2 - x1, y2 - y1)

        let labelText = '未识别'
        if (obj.type === 'recognized') {
          labelText = `${obj.reagent_name} ${obj.confidence_pct}`
        } else if (obj.best_candidate_name) {
          labelText = `${obj.best_candidate_name} ${obj.confidence_pct}`
        } else if (obj.best_candidate) {
          labelText = `${obj.best_candidate} ${obj.confidence_pct}`
        }

        let labelY = y1 - labelHeight
        let labelBelow = false

        if (labelY < 0) {
          labelY = y2
          labelBelow = true
        }

        const newLabelRect = {
          x: x1,
          y: labelY,
          width: labelWidth,
          height: labelHeight
        }

        let hasOverlap = false
        for (const existingRect of labelPositions) {
          if (rectsOverlap(newLabelRect, existingRect)) {
            hasOverlap = true
            break
          }
        }

        if (hasOverlap) {
          labelY = y2
          labelBelow = true

          const belowLabelRect = {
            x: x1,
            y: labelY,
            width: labelWidth,
            height: labelHeight
          }

          let stillOverlapping = false
          for (const existingRect of labelPositions) {
            if (rectsOverlap(belowLabelRect, existingRect)) {
              stillOverlapping = true
              break
            }
          }

          if (stillOverlapping) {
            const boxHeight = y2 - y1
            const centerY = y1 + boxHeight / 2
            labelY = centerY - labelHeight / 2
          }
        }

        ctx.fillStyle = color
        ctx.fillRect(x1, labelY, labelWidth, labelHeight)
        ctx.fillStyle = '#fff'
        ctx.font = '14px Arial'
        ctx.fillText(labelText, x1 + labelPadding, labelY + 18)

        labelPositions.push({
          x: x1,
          y: labelY,
          width: labelWidth,
          height: labelHeight
        })
      })
    }
    img.onerror = () => {
      console.error('Failed to load image for drawing bounding boxes')
    }
    img.src = capturedImage
  }

  const rectsOverlap = (rect1, rect2) => {
    const padding = 4
    return !(
      rect1.x + rect1.width + padding < rect2.x ||
      rect2.x + rect2.width + padding < rect1.x ||
      rect1.y + rect1.height + padding < rect2.y ||
      rect2.y + rect2.height + padding < rect1.y
    )
  }

  useEffect(() => {
    if (capturedImage && result) {
      drawBoundingBoxes()
    }
  }, [capturedImage, result])

  const confidenceColor = (score) => {
    if (!score) return '#d9d9d9'
    if (score >= 0.9) return '#52c41a'
    if (score >= 0.75) return '#1890ff'
    return '#ff4d4f'
  }

  const clampConfidence = (confidence) => {
    return Math.min(Math.max(confidence, 0), 1)
  }

  const handleOpenCorrection = (obj, type) => {
    setCorrectionObject({ obj, type })
    setShowCorrectionModal(true)
    setCropPixels(null)
    form.setFieldsValue({
      original_recognition_id: type === 'recognized' ? obj.reagent_id : (obj.best_candidate || ''),
      original_confidence: obj.confidence,
    })
  }

  const handleSubmitCorrection = async (values) => {
    setSubmittingCorrection(true)
    try {
      let reagentId = values.corrected_reagent_id
      let reagentName = values.corrected_reagent_name

      if (isNewReagentMode) {
        setCreatingReagent(true)
        try {
          const reagentData = {
            reagent_id: values.new_reagent_id || undefined,
            reagent_name: values.new_reagent_name,
            cas_number: values.cas_number,
            manufacturer: values.manufacturer,
            batch_number: values.batch_number,
            expiry_date: values.expiry_date,
            location: values.location,
            notes: values.notes,
          }
          const createRes = await api.createReagent(reagentData)
          if (createRes.success) {
            reagentId = createRes.reagent_id
            reagentName = values.new_reagent_name
            message.success(`新试剂 ${reagentId} 创建成功！`)
            await loadReagents()
          } else {
            throw new Error('创建试剂失败')
          }
        } catch (e) {
          message.error('创建试剂失败：' + (e.response?.data?.detail || e.message))
          return
        } finally {
          setCreatingReagent(false)
        }
      }

      const b64 = capturedImage.split(',')[1]
      const blob = await fetch(`data:image/jpeg;base64,${b64}`).then(res => res.blob())

      const formData = new FormData()
      formData.append('file', blob, 'correction.jpg')
      formData.append('corrected_reagent_id', reagentId)
      formData.append('corrected_reagent_name', reagentName)
      // 多物体纠错：优先用检测框进行裁剪，避免背景干扰
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
      formData.append('correction_source', 'multiple_recognize_page')

      const res = await api.submitCorrection(formData)
      if (res.success) {
        message.success('纠错提交成功！已自动应用到识别系统')
        setShowCorrectionModal(false)
        form.resetFields()
        setCorrectionObject(null)
        setIsNewReagentMode(false)
      }
    } catch (e) {
      message.error('提交失败：' + (e.response?.data?.detail || e.message))
    } finally {
      setSubmittingCorrection(false)
    }
  }

  return (
    <Row gutter={24}>
      {/* 左侧：摄像头和控制 */}
      <Col span={14}>
        <Card
          title="多试剂检测"
          extra={
            <Space>
              {/*<Button*/}
              {/*  icon={<SettingOutlined />}*/}
              {/*  onClick={() => setShowSettings(true)}*/}
              {/*>*/}
              {/*  设置*/}
              {/*</Button>*/}
            </Space>
          }
        >
          {/* 摄像头/图片区域 */}
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
            <canvas
              ref={canvasRef}
              style={{
                width: '100%',
                display: !cameraActive && capturedImage ? 'block' : 'none',
              }}
            />
            {!cameraActive && !capturedImage && (
              <div style={{ color: '#555', textAlign: 'center' }}>
                <CameraOutlined style={{ fontSize: 64, color: '#333' }} />
                <div style={{ color: '#888', marginTop: 8 }}>开启摄像头</div>
              </div>
            )}
            {loading && (
              <div style={{
                position: 'absolute', top: 0, left: 0, right: 0, bottom: 0,
                background: 'rgba(0,0,0,0.5)',
                display: 'flex', alignItems: 'center', justifyContent: 'center',
              }}>
                <Spin size="large" tip="检测中..." />
              </div>
            )}
          </div>

          {/* 控制按钮 */}
          <div style={{ marginTop: 16, display: 'flex', gap: 12, justifyContent: 'center' }}>
            {cameraActive ? (
              <>
                <Button
                  type="primary"
                  size="large"
                  icon={<SearchOutlined />}
                  onClick={captureAndRecognize}
                  disabled={loading}
                  loading={loading}
                >
                  立即检测
                </Button>
                <Button
                  danger
                  size="large"
                  onClick={stopCamera}
                >
                  关闭摄像头
                </Button>
              </>
            ) : (
              <>
                <Button
                  type="primary"
                  size="large"
                  icon={<CameraOutlined />}
                  onClick={startCamera}
                >
                  开启摄像头
                </Button>
                <Upload
                  accept="image/*"
                  showUploadList={false}
                  beforeUpload={handleUpload}
                >
                  <Button
                    size="large"
                    icon={<UploadOutlined />}
                    disabled={loading}
                  >
                    上传图片
                  </Button>
                </Upload>
              </>
            )}
            {cameraActive && (
              <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                <span style={{ color: '#666' }}>自动检测</span>
                <Switch
                  checked={autoRecognize}
                  onChange={setAutoRecognize}
                  checkedChildren="开"
                  unCheckedChildren="关"
                />
              </div>
            )}
          </div>
        </Card>
      </Col>

      {/* 右侧：检测结果 */}
      <Col span={10}>
        {/* 统计信息 */}
        {/*<Card style={{ marginBottom: 16 }}>*/}
        {/*  <Row gutter={16}>*/}
        {/*    <Col span={8}>*/}
        {/*      <Statistic*/}
        {/*        title="检测总数"*/}
        {/*        value={result?.total_objects || 0}*/}
        {/*        prefix={<AppstoreOutlined />}*/}
        {/*        valueStyle={{ color: '#1890ff' }}*/}
        {/*      />*/}
        {/*    </Col>*/}
        {/*    <Col span={8}>*/}
        {/*      <Statistic*/}
        {/*        title="识别成功"*/}
        {/*        value={result?.recognized_count || 0}*/}
        {/*        prefix={<CheckCircleOutlined />}*/}
        {/*        valueStyle={{ color: '#52c41a' }}*/}
        {/*      />*/}
        {/*    </Col>*/}
        {/*    <Col span={8}>*/}
        {/*      <Statistic*/}
        {/*        title="未识别"*/}
        {/*        value={result?.unrecognized_count || 0}*/}
        {/*        prefix={<CloseCircleOutlined />}*/}
        {/*        valueStyle={{ color: '#ff4d4f' }}*/}
        {/*      />*/}
        {/*    </Col>*/}
        {/*  </Row>*/}
        {/*</Card>*/}

        {/* 识别结果列表 */}
        <Card
          title="识别结果"
          bodyStyle={{ padding: '12px', maxHeight: 500, overflowY: 'auto' }}
        >
          {!result ? (
            <div style={{ textAlign: 'center', padding: '32px 0', color: '#999' }}>
              <ExperimentOutlined style={{ fontSize: 48 }} />
              <div style={{ marginTop: 8 }}>等待检测...</div>
            </div>
          ) : (
            <>
              <Alert
                message={result.message}
                type={result.recognized_count > 0 ? 'success' : 'warning'}
                style={{ marginTop: 12 }}
                showIcon
              />
              {result.recognized_objects?.length > 0 && (
                <>
                  <Divider orientation="left" style={{ margin: '12px 0 8px 0' }}>已识别</Divider>
                  {result.recognized_objects.map((obj, idx) => (
                    <Card
                      key={idx}
                      size="small"
                      style={{ marginBottom: 8, borderLeft: '3px solid #52c41a' }}
                      bodyStyle={{ padding: '8px 12px' }}
                    >
                      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 4 }}>
                        <strong style={{ fontSize: 13 }}>{obj.reagent_name || obj.reagent_id}</strong>
                        <Tag color="green" style={{ fontSize: 11, margin: 0 }}>{obj.confidence_pct}</Tag>
                      </div>
                      {obj.reagent_id && obj.reagent_id !== obj.reagent_name && (
                        <div style={{ color: '#999', fontSize: 11, marginBottom: 4 }}>
                          ID: {obj.reagent_id}
                        </div>
                      )}
                      <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 4 }}>
                        <Progress
                          percent={Math.round(clampConfidence(obj.confidence) * 100)}
                          strokeColor={confidenceColor(obj.confidence)}
                          showInfo={false}
                          size="small"
                          style={{ flex: 1 }}
                        />
                        <span style={{ fontSize: 11, color: confidenceColor(obj.confidence), minWidth: 40, textAlign: 'right' }}>
                          {clampConfidence(obj.confidence).toFixed(2)}
                        </span>
                      </div>
                      <Button
                        size="small"
                        type="link"
                        icon={<EditOutlined />}
                        onClick={() => handleOpenCorrection(obj, 'recognized')}
                        style={{ padding: 0, height: 'auto', fontSize: 11 }}
                      >
                        识别错误？纠错
                      </Button>
                    </Card>
                  ))}
                </>
              )}

              {result.unrecognized_objects?.length > 0 && (
                <>
                  <Divider orientation="left" style={{ margin: '12px 0 8px 0' }}>未识别</Divider>
                  {result.unrecognized_objects.map((obj, idx) => (
                    <Card
                      key={idx}
                      size="small"
                      style={{ marginBottom: 8, borderLeft: '3px solid #ff4d4f' }}
                      bodyStyle={{ padding: '8px 12px' }}
                    >
                      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 4 }}>
                        <strong style={{ fontSize: 13, color: '#ff4d4f' }}>未知物体</strong>
                        <Tag color="red" style={{ fontSize: 11, margin: 0 }}>{obj.confidence_pct}</Tag>
                      </div>
                      {obj.best_candidate && (
                        <div style={{ color: '#999', fontSize: 11, marginBottom: 4 }}>
                          最相似: {obj.best_candidate_name || obj.best_candidate} {obj.best_candidate_name && obj.best_candidate_name !== obj.best_candidate && `(${obj.best_candidate})`}
                        </div>
                      )}
                      <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 4 }}>
                        <Progress
                          percent={Math.round(clampConfidence(obj.confidence) * 100)}
                          strokeColor={confidenceColor(obj.confidence)}
                          showInfo={false}
                          size="small"
                          style={{ flex: 1 }}
                        />
                        <span style={{ fontSize: 11, color: confidenceColor(obj.confidence), minWidth: 40, textAlign: 'right' }}>
                          {clampConfidence(obj.confidence).toFixed(2)}
                        </span>
                      </div>
                      <Button
                        size="small"
                        type="link"
                        danger
                        icon={<EditOutlined />}
                        onClick={() => handleOpenCorrection(obj, 'unrecognized')}
                        style={{ padding: 0, height: 'auto', fontSize: 11 }}
                      >
                        点击纠错
                      </Button>
                    </Card>
                  ))}
                </>
              )}
            </>
          )}
        </Card>

        {/* 检测历史 */}
        <Card
          title="检测记录"
          style={{ marginTop: 16 }}
          bodyStyle={{ padding: '8px 16px', maxHeight: 300, overflowY: 'auto' }}
        >
          {history.length === 0 ? (
            <div style={{ textAlign: 'center', padding: '24px 0', color: '#999' }}>
              暂无记录
            </div>
          ) : (
            <List
              dataSource={history}
              renderItem={(item) => (
                <List.Item
                  style={{ cursor: 'pointer', padding: '8px 0' }}
                  onClick={() => {
                    setCapturedImage(item.image)
                    setResult(item.result)
                  }}
                >
                  <List.Item.Meta
                    avatar={<Avatar icon={<AppstoreOutlined />} style={{ background: '#1890ff' }} />}
                    title={
                      <Space>
                        <span>检测到 {item.total} 个物体</span>
                        <Tag color="green">识别 {item.recognized}</Tag>
                        <Tag color="red">未识别 {item.unrecognized}</Tag>
                      </Space>
                    }
                    description={item.timestamp}
                  />
                  <img
                    src={item.image}
                    alt=""
                    style={{ width: 60, height: 60, objectFit: 'cover', borderRadius: 4 }}
                  />
                </List.Item>
              )}
            />
          )}
        </Card>
      </Col>

      {/* 设置抽屉 */}
      <Drawer
        title="检测设置"
        placement="right"
        width={300}
        onClose={() => setShowSettings(false)}
        open={showSettings}
      >
        <Space direction="vertical" style={{ width: '100%' }} size={24}>
          <div>
            <div style={{ marginBottom: 8 }}>
              <strong>检测置信度阈值</strong>
              <div style={{ color: '#999', fontSize: 12 }}>
                较低的值会检测更多物体，但可能增加误检
              </div>
            </div>
            <Slider
              min={0.1}
              max={0.9}
              step={0.1}
              value={minConfidence}
              onChange={setMinConfidence}
              marks={{
                0.1: '0.1',
                0.5: '0.5',
                0.9: '0.9',
              }}
            />
            <div style={{ textAlign: 'center', marginTop: 8 }}>
              <Tag color="blue">当前: {minConfidence}</Tag>
            </div>
          </div>

          <Divider />

          <div>
            <strong>使用说明</strong>
            <ul style={{ marginTop: 8, paddingLeft: 16, color: '#666', fontSize: 13 }}>
              <li>开启摄像头实时检测</li>
              {/*<li>或上传图片进行检测</li>*/}
              <li>可调整检测阈值</li>
              <li>支持自动连续检测</li>
              <li>点击历史记录查看详情</li>
            </ul>
          </div>
        </Space>
      </Drawer>

      {/* 纠错弹窗 */}
      <Modal
        title="纠错识别结果"
        open={showCorrectionModal}
        onCancel={() => {
          setShowCorrectionModal(false)
          form.resetFields()
          setCorrectionObject(null)
          setCropPixels(null)
          setIsNewReagentMode(false)
        }}
        afterClose={() => {
          setCorrectionObject(null)
          setCropPixels(null)
        }}
        footer={null}
        width={900}
        destroyOnClose={true}
      >
        <Form
          form={form}
          layout="vertical"
          onFinish={handleSubmitCorrection}
        >
          <Row gutter={24}>
            <Col span={12}>
              {capturedImage && correctionObject && (
                <div>
                  <ImageCropper
                    key={`cropper-${correctionObject?.type}-${correctionObject?.obj?.reagent_id || Date.now()}`}
                    src={capturedImage}
                    height={400}
                    initialCrop={correctionObject?.obj?.crop_bbox
                      ? {
                          x1: correctionObject.obj.crop_bbox[0],
                          y1: correctionObject.obj.crop_bbox[1],
                          x2: correctionObject.obj.crop_bbox[2],
                          y2: correctionObject.obj.crop_bbox[3],
                        }
                      : (correctionObject?.obj?.bbox
                        ? {
                            x1: correctionObject.obj.bbox[0],
                            y1: correctionObject.obj.bbox[1],
                            x2: correctionObject.obj.bbox[2],
                            y2: correctionObject.obj.bbox[3],
                          }
                        : null)}
                    onChange={(c) => setCropPixels(c)}
                  />
                  {correctionObject && (
                    <div style={{ marginTop: 8, fontSize: 12, color: '#666' }}>
                      检测区域: [{correctionObject.obj.bbox.join(', ')}]
                    </div>
                  )}
                </div>
              )}
            </Col>
            <Col span={12}>
              <Form.Item
                name="corrected_reagent_id"
                label="正确的试剂名称"
                rules={[{ required: !isNewReagentMode, message: '请选择试剂' }]}
              >
                <Select
                  showSearch
                  placeholder="按名称搜索/选择"
                  filterOption={(input, option) =>
                    (option?.label ?? '').toLowerCase().includes(input.toLowerCase())
                  }
                  onChange={(rid) => {
                    if (rid === '__new__') {
                      setIsNewReagentMode(true)
                    } else {
                      setIsNewReagentMode(false)
                      form.setFieldsValue({
                        corrected_reagent_name: reagents.find(x => x.reagent_id === rid)?.reagent_name || '',
                        new_reagent_id: undefined,
                        new_reagent_name: undefined,
                      })
                    }
                  }}
                  options={[
                    ...reagents.map(r => ({ label: r.reagent_name, value: r.reagent_id })),
                    { label: '+ 新增试剂', value: '__new__' }
                  ]}
                />
              </Form.Item>
              <Form.Item
                name="corrected_reagent_name"
                hidden
              >
                <Input />
              </Form.Item>

              {isNewReagentMode && (
                <>
                  <Alert
                    // message="新增试剂"
                    description="请填写新试剂的基本信息，系统将自动创建试剂并注册此图片。"
                    type="success"
                    showIcon={false}
                    style={{ marginBottom: 16 }}
                  />
                  {/*<Form.Item*/}
                  {/*  name="new_reagent_id"*/}
                  {/*  label="试剂ID（可选）"*/}
                  {/*  extra="留空则自动生成（如：乙醇001）"*/}
                  {/*>*/}
                  {/*  <Input placeholder="如：乙醇001" />*/}
                  {/*</Form.Item>*/}
                  <Form.Item
                    name="new_reagent_name"
                    label="试剂名称"
                    rules={[{ required: true, message: '请输入试剂名称' }]}
                  >
                    <Input placeholder="如：乙醇" />
                  </Form.Item>
                </>
              )}
              <Form.Item
                name="original_recognition_id"
                label="原识别结果（可选）"
              >
                <Input placeholder="如：乙醇002" />
              </Form.Item>
              <Form.Item
                name="notes"
                label="备注（可选）"
              >
                <Input.TextArea rows={2} placeholder="如：标签模糊导致误识别" />
              </Form.Item>
              <Form.Item
                name="apply_immediately"
                initialValue={true}
                label="是否立即应用"
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
            </Col>
          </Row>
        </Form>
      </Modal>
    </Row>
  )
}