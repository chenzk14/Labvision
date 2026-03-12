import React, { useRef, useState, useEffect } from "react"
import ReactCrop from "react-image-crop"
import "react-image-crop/dist/ReactCrop.css"

export default function ImageCropper({
  src,
  initialCrop,
  onChange,
  height = 260
}) {
  const imgRef = useRef(null)
  const previewCanvasRef = useRef(null)

  const [crop, setCrop] = useState()
  const [completedCrop, setCompletedCrop] = useState()

  // 图片加载
  const onImageLoad = (e) => {
    const img = e.currentTarget
    imgRef.current = img

    if (initialCrop) {
      setCrop({
        unit: "px",
        x: initialCrop.x1,
        y: initialCrop.y1,
        width: initialCrop.x2 - initialCrop.x1,
        height: initialCrop.y2 - initialCrop.y1
      })
    } else {
      setCrop({
        unit: "%",
        x: 10,
        y: 10,
        width: 80,
        height: 80
      })
    }
  }

  // 输出原图坐标
  useEffect(() => {
    if (!completedCrop || !imgRef.current) return

    const image = imgRef.current

    const scaleX = image.naturalWidth / image.width
    const scaleY = image.naturalHeight / image.height

    const x1 = Math.round(completedCrop.x * scaleX)
    const y1 = Math.round(completedCrop.y * scaleY)
    const x2 = Math.round((completedCrop.x + completedCrop.width) * scaleX)
    const y2 = Math.round((completedCrop.y + completedCrop.height) * scaleY)

    onChange?.(
      {
        x1,
        y1,
        x2,
        y2,
        w: x2 - x1,
        h: y2 - y1
      },
      {
        naturalWidth: image.naturalWidth,
        naturalHeight: image.naturalHeight
      }
    )

    drawPreview(image, completedCrop)
  }, [completedCrop, onChange])

  // 绘制裁剪预览
  const drawPreview = (image, crop) => {
    const canvas = previewCanvasRef.current
    if (!canvas) return

    const ctx = canvas.getContext("2d")

    const scaleX = image.naturalWidth / image.width
    const scaleY = image.naturalHeight / image.height

    canvas.width = crop.width
    canvas.height = crop.height

    ctx.drawImage(
      image,
      crop.x * scaleX,
      crop.y * scaleY,
      crop.width * scaleX,
      crop.height * scaleY,
      0,
      0,
      crop.width,
      crop.height
    )
  }

  // 获取裁剪图片
  const getCroppedImage = () => {
    const canvas = previewCanvasRef.current
    if (!canvas) return null
    return canvas.toDataURL("image/png")
  }

  return (
    <div style={{ width: "100%" }}>
      <ReactCrop
        crop={crop}
        keepSelection
        minWidth={20}
        minHeight={20}
        onChange={(c) => setCrop(c)}
        onComplete={(c) => setCompletedCrop(c)}
      >
        <img
          src={src}
          alt="crop"
          onLoad={onImageLoad}
          style={{
            maxHeight: height,
            width: "auto"
          }}
        />
      </ReactCrop>

      <div style={{ marginTop: 10 }}>
        <div style={{ fontSize: 12, marginBottom: 4 }}>
          裁剪预览
        </div>

        <canvas
          ref={previewCanvasRef}
          style={{
            border: "1px solid #ddd",
            maxWidth: "100%"
          }}
        />
      </div>

      <button
        style={{ marginTop: 10 }}
        onClick={() => {
          const img = getCroppedImage()
          if (img) {
            console.log("裁剪图片:", img)
          }
        }}
      >
        获取裁剪图片
      </button>
    </div>
  )
}