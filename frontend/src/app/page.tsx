"use client";

import { useState } from "react";
import Image from "next/image";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Separator } from "@/components/ui/separator";
import { Skeleton } from "@/components/ui/skeleton";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Upload, Banana, AlertCircle, CheckCircle } from "lucide-react";

type Detection = {
  label: string;
  confidence: number;
  box: number[];
  description: string;
  days_until_ripe?: number | null;
};

type PredictResponse = {
  // Image response
  banana_found?: boolean;
  count?: number;
  results?: Detection[];
  annotated_image?: string | null;

  // Video response
  frame_count_analyzed?: number;
  banana_found_in_any_frame?: boolean;
  most_common_label?: string | null;
  frame_results?: {
    timestamp: number;
    detections: Detection[];
  }[];
  best_frame_annotated_image?: string | null;
};

export default function Home() {
  const [preview, setPreview] = useState<string | null>(null);
  const [fileName, setFileName] = useState<string | null>(null);
  const [fileType, setFileType] = useState<"image" | "video" | null>(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<PredictResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  async function handleImageAsync(file: File) {
    setPreview(URL.createObjectURL(file));
    setFileName(file.name);
    setFileType("image");
    setResult(null);
    setError(null);
    setLoading(true);

    const formData = new FormData();
    formData.append("file", file);

    try {
      const res = await fetch("http://localhost:8000/predict", {
        method: "POST",
        body: formData,
      });
      const data = await res.json();
      if (!res.ok) {
        setError(data.detail || "Something went wrong while analyzing the image.");
      } else {
        setResult(data);
      }
    } catch {
      setError("Could not reach the server. Is the backend running?");
    } finally {
      setLoading(false);
    }
  }

  async function handleVideoAsync(file: File) {
    setPreview(URL.createObjectURL(file));
    setFileName(file.name);
    setFileType("video");
    setResult(null);
    setError(null);
    setLoading(true);

    const formData = new FormData();
    formData.append("file", file);

    try {
      const res = await fetch("http://localhost:8000/predict-video", {
        method: "POST",
        body: formData,
      });
      const data = await res.json();
      if (!res.ok) {
        setError(data.detail || "Something went wrong while analyzing the video.");
      } else {
        setResult(data);
      }
    } catch {
      setError("Could not reach the server. Is the backend running?");
    } finally {
      setLoading(false);
    }
  }

  function onInputChange(e: React.ChangeEvent<HTMLInputElement>) {
    const file = e.target.files?.[0];
    if (!file) return;

    if (file.type.startsWith("image/")) {
      handleImageAsync(file);
    } else if (file.type.startsWith("video/")) {
      handleVideoAsync(file);
    } else {
      setError("Please upload an image or video file.");
    }
  }

  const isImageResult = fileType === "image";
  const isVideoResult = fileType === "video";

  return (
    <main className="flex min-h-screen items-center justify-center bg-gray-50 p-8">
      <div className="flex w-full max-w-5xl gap-8">
        {/* Left panel: upload + preview */}
        <Card className="flex flex-1 items-center justify-center border-dashed p-4">
          <CardContent className="w-full p-4">
            {preview ? (
              <div className="w-full text-center">
                <div className="relative mx-auto h-56 w-full overflow-hidden rounded-md bg-black/5">
                  {isVideoResult ? (
                    <video
                      src={preview}
                      controls
                      className="h-full w-full object-contain"
                    />
                  ) : (
                    <Image
                      src={preview}
                      alt="Uploaded file preview"
                      fill
                      className="object-contain"
                      unoptimized
                    />
                  )}
                </div>
                {fileName && (
                  <p className="mt-2 truncate text-xs text-gray-500">{fileName}</p>
                )}
                <label
                  htmlFor="file-upload"
                  className="relative mt-4 inline-block cursor-pointer text-sm font-semibold text-blue-600 hover:text-blue-500"
                >
                  Choose a different file
                  <input
                    id="file-upload"
                    name="file-upload"
                    type="file"
                    className="sr-only"
                    accept=".jpg, .jpeg, .png, .mp4, .mov, .webm"
                    onChange={onInputChange}
                  />
                </label>
              </div>
            ) : (
              <div className="text-center">
                <Upload className="mx-auto mb-3 h-8 w-8 text-gray-400" />
                <label
                  htmlFor="file-upload"
                  className="relative cursor-pointer rounded-md bg-blue-600 px-4 py-2 text-sm font-semibold text-white shadow-sm hover:bg-blue-500 focus-within:outline-none focus-within:ring-2 focus-within:ring-blue-600 focus-within:ring-offset-2"
                >
                  <span>Upload an image or video</span>
                  <input
                    id="file-upload"
                    name="file-upload"
                    type="file"
                    className="sr-only"
                    accept=".jpg, .jpeg, .png, .mp4, .mov, .webm"
                    onChange={onInputChange}
                  />
                </label>
                <p className="mt-2 text-xs text-gray-500">PNG, JPG, MP4, MOV, or WEBM</p>
              </div>
            )}
          </CardContent>
        </Card>

        {/* Right panel: results */}
        <Card className="flex flex-1 flex-col border-dashed p-4">
          <CardHeader>
            <CardTitle className="flex items-center gap-2 text-base">
              <Banana className="h-4 w-4" /> Results
            </CardTitle>
            <CardDescription>
              {isVideoResult ? "Frame-by-frame video analysis" : "Image analysis"}
            </CardDescription>
          </CardHeader>
          <CardContent className="flex-1">
            {loading && (
              <div className="space-y-3">
                <p className="text-sm text-gray-500">
                  {isVideoResult ? "Analyzing video frames..." : "Analyzing image..."}
                </p>
                <Skeleton className="h-40 w-full" />
                <Skeleton className="h-4 w-3/4" />
                <Skeleton className="h-4 w-1/2" />
              </div>
            )}

            {error && (
              <Alert variant="destructive">
                <AlertCircle className="h-4 w-4" />
                <AlertTitle>Error</AlertTitle>
                <AlertDescription>{error}</AlertDescription>
              </Alert>
            )}

            {/* IMAGE RESULTS */}
            {!loading && !error && isImageResult && result && result.banana_found && (
              <div className="w-full space-y-4">
                {result.annotated_image && (
                  <div className="relative mx-auto h-56 w-full overflow-hidden rounded-md">
                    <Image
                      src={`data:image/png;base64,${result.annotated_image}`}
                      alt="Detected banana with bounding box"
                      fill
                      className="object-contain"
                      unoptimized
                    />
                  </div>
                )}
                {result.results?.map((detection, i) => (
                  <div key={i} className="rounded-md border border-gray-200 p-3">
                    <div className="flex items-center justify-between">
                      <p className="text-sm font-semibold capitalize text-gray-900">
                        {detection.label}
                      </p>
                      <Badge variant="secondary">
                        {Math.round(detection.confidence * 100)}%
                      </Badge>
                    </div>
                    <p className="mt-1 text-sm text-gray-600">{detection.description}</p>
                    {detection.days_until_ripe != null && (
                      <p className="mt-1 text-xs text-gray-500">
                        Ripe in ~{detection.days_until_ripe} day(s)
                      </p>
                    )}
                  </div>
                ))}
              </div>
            )}

            {!loading && !error && isImageResult && result && !result.banana_found && (
              <p className="text-sm text-gray-500">No banana detected in that image.</p>
            )}

            {/* VIDEO RESULTS */}
            {!loading && !error && isVideoResult && result && (
              <div className="w-full space-y-4">
                <div className="flex flex-wrap items-center gap-2">
                  {result.banana_found_in_any_frame ? (
                    <Badge className="flex items-center gap-1 bg-green-600">
                      <CheckCircle className="h-3 w-3" /> Banana detected
                    </Badge>
                  ) : (
                    <Badge variant="secondary">No banana detected</Badge>
                  )}
                  {result.most_common_label && (
                    <Badge variant="outline">Most common: {result.most_common_label}</Badge>
                  )}
                  {typeof result.frame_count_analyzed === "number" && (
                    <span className="text-xs text-gray-500">
                      {result.frame_count_analyzed} frame(s) analyzed
                    </span>
                  )}
                </div>

                {result.best_frame_annotated_image && (
                  <div className="relative mx-auto h-56 w-full overflow-hidden rounded-md">
                    <Image
                      src={`data:image/png;base64,${result.best_frame_annotated_image}`}
                      alt="Best detected frame"
                      fill
                      className="object-contain"
                      unoptimized
                    />
                  </div>
                )}

                <Separator />

                <div className="max-h-64 space-y-3 overflow-y-auto pr-1">
                  {result.frame_results?.map((frame, i) => (
                    <div key={i} className="rounded-md border border-gray-200 p-3">
                      <p className="text-xs font-semibold text-gray-500">
                        t = {frame.timestamp}s
                      </p>
                      {frame.detections.length === 0 ? (
                        <p className="mt-1 text-sm text-gray-400">No detections</p>
                      ) : (
                        frame.detections.map((d, j) => (
                          <div key={j} className="mt-1 flex items-center justify-between">
                            <p className="text-sm capitalize text-gray-800">{d.label}</p>
                            <Badge variant="secondary">
                              {Math.round(d.confidence * 100)}%
                            </Badge>
                          </div>
                        ))
                      )}
                    </div>
                  ))}
                </div>
              </div>
            )}

            {!loading && !error && !result && (
              <p className="text-sm text-gray-400">
                Results will appear here after you upload a file.
              </p>
            )}
          </CardContent>
        </Card>
      </div>
    </main>
  );
}