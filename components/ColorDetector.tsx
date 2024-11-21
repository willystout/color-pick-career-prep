'use client'

import { useState } from 'react'

export default function ColorDetector() {
    const [color, setColor] = useState('#FFFFFF')
    const [prediction, setPrediction] = useState<string | null>(null)
    const [loading, setLoading] = useState(false)

    const detectColor = async () => {
        setLoading(true)
        try {
            const hexColor = color.replace('#', '')
            if (!hexColor.match(/^[0-9A-Fa-f]{6}$/i)) {
                throw new Error('Invalid color format')
            }
            if (hexColor.length !== 6) {
                throw new Error('Invalid color format')
            }
            const response = await fetch(`http://localhost:5000/identify?color=${hexColor}`)
            const data = await response.json()
            setPrediction(data.color_name)
        } catch (err) {
            console.error('Error:', err)
            setPrediction(null)
        } finally {
            setLoading(false)
        }
    }

    return (
        <div className="max-w-md mx-auto bg-white rounded-lg shadow p-6">
            <h1 className="text-2xl font-bold mb-4">Color Detector</h1>

            <div className="space-y-4">
                <div>
                    <label className="block text-sm font-medium mb-2">
                        Select a color:
                    </label>
                    <div className="flex gap-4">
                        <input
                            type="color"
                            value={color}
                            onChange={(e) => setColor(e.target.value)}
                            className="h-10 w-20"
                        />
                        <input
                            type="text"
                            value={color}
                            onChange={(e) => setColor(e.target.value)}
                            className="border rounded px-3 py-2 w-full"
                            placeholder="#FFFFFF"
                        />
                    </div>
                </div>

                <button
                    onClick={detectColor}
                    disabled={loading}
                    className="w-full bg-blue-600 text-white py-2 px-4 rounded hover:bg-blue-700 disabled:opacity-50"
                >
                    {loading ? 'Detecting...' : 'Detect Color'}
                </button>

                {prediction && (
                    <div className="mt-4">
                        <div className="text-lg font-semibold">Result:</div>
                        <div className="flex items-center gap-3 mt-2">
                            <div
                                className="w-6 h-6 rounded border"
                                style={{ backgroundColor: color }}
                            />
                            <span>{prediction}</span>
                        </div>
                    </div>
                )}
            </div>
        </div>
    )
}