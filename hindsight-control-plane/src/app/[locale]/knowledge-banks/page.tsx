import { redirect } from "next/navigation";

// Knowledge banks are listed with the memory banks, under the switch on the overview —
// this path stays only so an old link still lands somewhere sensible.
export const dynamic = "force-dynamic";

export default function KnowledgeBanksPage() {
  redirect("/dashboard?kind=knowledge");
}
