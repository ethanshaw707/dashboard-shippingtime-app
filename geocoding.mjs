import fs from "node:fs";

if (typeof fetch !== "function") {
  throw new Error("This script requires Node.js 18+ (global fetch is missing).");
}

const addresses = [
  ["123 Flatbush Ave","Brooklyn","NY","11217"],
  ["456 Desert Bloom Rd","Phoenix","AZ","85004"],
  ["789 Bayou Park Dr","Houston","TX","77002"],
  ["210 Alamo Plaza Way","San Antonio","TX","78205"],
  ["135 River City Ln","Jacksonville","FL","32202"],
  ["246 Windy City Blvd","Chicago","IL","60601"],
  ["357 Lone Star Pkwy","Dallas","TX","75201"],
  ["468 Brandywine St","Wilmington","DE","19801"],
  ["579 Barton Springs Rd","Austin","TX","78704"],
  ["681 Hall of Fame Dr","Canton","OH","44702"],
  ["792 Queen City Cir","Charlotte","NC","28202"],
  ["101 Midtown Ave","New York","NY","10001"],
  ["212 Peachtree Center Ave","Atlanta","GA","30303"],
  ["324 Heritage Way","Franklin","TN","37064"],
  ["435 Bluegrass Pkwy","Lexington","KY","40507"],
  ["546 Sunbelt Dr","Orlando","FL","32801"],
  ["657 Flower City Blvd","Rochester","NY","14604"],
  ["768 Neon Strip Rd","Las Vegas","NV","89109"],
  ["879 Derby Gate St","Louisville","KY","40202"],
  ["981 Historic Sq","Marietta","GA","30060"],
  ["112 Pikes Peak View Rd","Colorado Springs","CO","80903"],
  ["224 Buckeye Ave","Columbus","OH","43215"],
  ["336 Lakeside Dr","Marble Falls","TX","78654"],
  ["448 Bayshore Blvd","Tampa","FL","33606"],
  ["559 Pacific Crest Rd","San Diego","CA","92101"],
  ["661 Capital View Dr","Columbia","SC","29201"],
  ["772 Main Street Ext","Greenville","SC","29601"],
  ["884 Cedar Grove Rd","Lebanon","TN","37087"],
  ["996 Willamette Way","Salem","OR","97301"],
  ["108 Rock Hall Ln","Cleveland","OH","44114"],
  ["219 Mile High Blvd","Denver","CO","80202"],
  ["331 Heartland Pkwy","Omaha","NE","68102"],
  ["442 Rose City Ave","Portland","OR","97205"],
  ["553 Oak City Dr","Raleigh","NC","27601"],
  ["665 Riverfront St","Richmond","VA","23219"],
  ["776 Magnolia Row","Jackson","MS","39201"],
  ["887 Empire State Plaza Way","Albany","NY","12207"],
  ["998 Stockyards Blvd","Fort Worth","TX","76164"],
  ["110 Thunder Way","Oklahoma City","OK","73102"],
  ["221 Tidewater Dr","Chesapeake","VA","23320"],
  ["332 Biscayne Bay Ave","Miami","FL","33131"],
  ["443 Harbor View Rd","Milford","CT","06460"],
  ["554 Steel City Pkwy","Pittsburgh","PA","15222"],
  ["665 Market View St","San Francisco","CA","94103"],
  ["776 Silicon Valley Blvd","San Jose","CA","95112"],
  ["887 Sonoran Vista Rd","Tucson","AZ","85701"],
  ["998 Atlantic Shore Dr","Virginia Beach","VA","23451"],
  ["109 Sandia Crest Way","Albuquerque","NM","87102"],
  ["210 Desert Willow Ln","Gilbert","AZ","85233"],
  ["321 Lincoln Parkway","Springfield","IL","62701"],
  ["432 Old Town Sq","Alexandria","VA","22314"],
  ["543 Heart of America Blvd","Kansas City","MO","64106"],
  ["654 Volunteer Way","Knoxville","TN","37902"],
  ["765 Green Mountain Dr","Lakewood","CO","80226"],
  ["876 Music Row Ave","Nashville","TN","37203"],
  ["987 Gateway Arch Dr","Saint Louis","MO","63101"],
  ["109 Puget Sound Ave","Seattle","WA","98101"],
];

async function geocode(q) {
  const url = new URL("https://nominatim.openstreetmap.org/search");
  url.searchParams.set("format", "jsonv2");
  url.searchParams.set("limit", "1");
  url.searchParams.set("q", q);

  const res = await fetch(url, {
    headers: {
      "User-Agent": "demo-map-geocoder/1.0 (your-email@example.com)"
    }
  });
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  const data = await res.json();
  return data[0] ? { lat: Number(data[0].lat), lng: Number(data[0].lon) } : null;
}

const out = [];
for (const [street, city, state, zip] of addresses) {
  const q = `${street}, ${city}, ${state} ${zip}, USA`;
  try {
    const r = await geocode(q);
    out.push({ street, city, state, zip, ...r });
    console.log(`${city}, ${state} ${zip} -> ${r ? `${r.lat}, ${r.lng}` : "no match"}`);
  } catch (e) {
    out.push({ street, city, state, zip, error: String(e) });
    console.log(`${city}, ${state} ${zip} -> error: ${String(e)}`);
  }
  // be nice to the free service
  await new Promise(r => setTimeout(r, 1100));
}

fs.writeFileSync("geocoded.json", JSON.stringify(out, null, 2));
console.log("Wrote geocoded.json");
