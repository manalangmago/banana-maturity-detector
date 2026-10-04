import { NextRequest, NextResponse } from "next/server";
 
const BACKEND_URL = process.env.BACKEND_URL || "http://localhost:8000";
 
export async function POST(req: NextRequest) {
  try {
    const formData = await req.formData();
    const file = formData.get("file");
 
    if (!file) {
      return NextResponse.json({ error: "No file uploaded" }, { status: 400 });
    }
 
    // Re-package the file into a new FormData to forward to Python
    const forwardData = new FormData();
    forwardData.append("file", file);
 
    const backendResponse = await fetch(`${BACKEND_URL}/predict-video`, {
      method: "POST",
      body: forwardData,
    });
 
    if (!backendResponse.ok) {
      const errorText = await backendResponse.text();
      return NextResponse.json(
        { error: `Backend error: ${errorText}` },
        { status: backendResponse.status }
      );
    }
 
    const result = await backendResponse.json();
    return NextResponse.json(result);
  } catch (err) {
    console.error(err);
    return NextResponse.json({ error: "Something went wrong" }, { status: 500 });
  }
}