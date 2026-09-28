import { NextRequest, NextResponse } from "next/server";
import { DATAPLANE_URL, getDataplaneHeaders } from "@/lib/hindsight-client";

// Knowledge banks are their own dataplane resource (/v1/default/knowledge-banks). The
// generated SDK does not cover them yet, so this forwards the request as-is; one route
// for the whole tree rather than one file per endpoint.
async function forward(request: NextRequest, { params }: { params: Promise<{ path?: string[] }> }) {
  const { path = [] } = await params;
  const suffix = path.map(encodeURIComponent).join("/");
  const url = `${DATAPLANE_URL}/v1/default/knowledge-banks${suffix ? `/${suffix}` : ""}${request.nextUrl.search}`;
  const contentType = request.headers.get("content-type") || "";
  // A file upload is multipart: forward the parts and let fetch build its own boundary,
  // because re-declaring Content-Type here would send the browser's boundary with a
  // differently-encoded body.
  const multipart = contentType.startsWith("multipart/form-data");
  const body = ["GET", "DELETE"].includes(request.method)
    ? undefined
    : multipart
      ? await request.formData()
      : await request.text();
  const response = await fetch(url, {
    method: request.method,
    headers: multipart
      ? getDataplaneHeaders()
      : getDataplaneHeaders({ "Content-Type": "application/json" }),
    body,
  });
  const text = await response.text();
  return new NextResponse(text, {
    status: response.status,
    headers: { "Content-Type": response.headers.get("Content-Type") || "application/json" },
  });
}

export { forward as GET, forward as POST, forward as PUT, forward as DELETE };
