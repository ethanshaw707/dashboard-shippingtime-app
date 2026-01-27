import csv
import re


INPUT_PAGE3_PATH = "page3.csv"
OUTPUT_PAGE4_PATH = "page4.csv"

ADDRESSES_RAW = """
Address("", "", "", "123 Flatbush Ave", "Brooklyn", "NY", "11217", address_residential_indicator="yes"),
Address("", "", "", "456 Desert Bloom Rd", "Phoenix", "AZ", "85004", address_residential_indicator="yes"),
Address("", "", "", "789 Bayou Park Dr", "Houston", "TX", "77002", address_residential_indicator="yes"),
Address("", "", "", "210 Alamo Plaza Way", "San Antonio", "TX", "78205", address_residential_indicator="yes"),
Address("", "", "", "135 River City Ln", "Jacksonville", "FL", "32202", address_residential_indicator="yes"),
Address("", "", "", "246 Windy City Blvd", "Chicago", "IL", "60601", address_residential_indicator="yes"),
Address("", "", "", "357 Lone Star Pkwy", "Dallas", "TX", "75201", address_residential_indicator="yes"),
Address("", "", "", "468 Brandywine St", "Wilmington", "DE", "19801", address_residential_indicator="yes"),
Address("", "", "", "579 Barton Springs Rd", "Austin", "TX", "78704", address_residential_indicator="yes"),
Address("", "", "", "681 Hall of Fame Dr", "Canton", "OH", "44702", address_residential_indicator="yes"),
Address("", "", "", "792 Queen City Cir", "Charlotte", "NC", "28202", address_residential_indicator="yes"),
Address("", "", "", "101 Midtown Ave", "New York", "NY", "10001", address_residential_indicator="yes"),
Address("", "", "", "212 Peachtree Center Ave", "Atlanta", "GA", "30303", address_residential_indicator="yes"),
Address("", "", "", "324 Heritage Way", "Franklin", "TN", "37064", address_residential_indicator="yes"),
Address("", "", "", "435 Bluegrass Pkwy", "Lexington", "KY", "40507", address_residential_indicator="yes"),
Address("", "", "", "546 Sunbelt Dr", "Orlando", "FL", "32801", address_residential_indicator="yes"),
Address("", "", "", "657 Flower City Blvd", "Rochester", "NY", "14604", address_residential_indicator="yes"),
Address("", "", "", "768 Neon Strip Rd", "Las Vegas", "NV", "89109", address_residential_indicator="yes"),
Address("", "", "", "879 Derby Gate St", "Louisville", "KY", "40202", address_residential_indicator="yes"),
Address("", "", "", "981 Historic Sq", "Marietta", "GA", "30060", address_residential_indicator="yes"),
Address("", "", "", "112 Pikes Peak View Rd", "Colorado Springs", "CO", "80903", address_residential_indicator="yes"),
Address("", "", "", "224 Buckeye Ave", "Columbus", "OH", "43215", address_residential_indicator="yes"),
Address("", "", "", "336 Lakeside Dr", "Marble Falls", "TX", "78654", address_residential_indicator="yes"),
Address("", "", "", "448 Bayshore Blvd", "Tampa", "FL", "33606", address_residential_indicator="yes"),
Address("", "", "", "559 Pacific Crest Rd", "San Diego", "CA", "92101", address_residential_indicator="yes"),
Address("", "", "", "661 Capital View Dr", "Columbia", "SC", "29201", address_residential_indicator="yes"),
Address("", "", "", "772 Main Street Ext", "Greenville", "SC", "29601", address_residential_indicator="yes"),
Address("", "", "", "884 Cedar Grove Rd", "Lebanon", "TN", "37087", address_residential_indicator="yes"),
Address("", "", "", "996 Willamette Way", "Salem", "OR", "97301", address_residential_indicator="yes"),
Address("", "", "", "108 Rock Hall Ln", "Cleveland", "OH", "44114", address_residential_indicator="yes"),
Address("", "", "", "219 Mile High Blvd", "Denver", "CO", "80202", address_residential_indicator="yes"),
Address("", "", "", "331 Heartland Pkwy", "Omaha", "NE", "68102", address_residential_indicator="yes"),
Address("", "", "", "442 Rose City Ave", "Portland", "OR", "97205", address_residential_indicator="yes"),
Address("", "", "", "553 Oak City Dr", "Raleigh", "NC", "27601", address_residential_indicator="yes"),
Address("", "", "", "665 Riverfront St", "Richmond", "VA", "23219", address_residential_indicator="yes"),
Address("", "", "", "776 Magnolia Row", "Jackson", "MS", "39201", address_residential_indicator="yes"),
Address("", "", "", "887 Empire State Plaza Way", "Albany", "NY", "12207", address_residential_indicator="yes"),
Address("", "", "", "998 Stockyards Blvd", "Fort Worth", "TX", "76164", address_residential_indicator="yes"),
Address("", "", "", "110 Thunder Way", "Oklahoma City", "OK", "73102", address_residential_indicator="yes"),
Address("", "", "", "221 Tidewater Dr", "Chesapeake", "VA", "23320", address_residential_indicator="yes"),
Address("", "", "", "332 Biscayne Bay Ave", "Miami", "FL", "33131", address_residential_indicator="yes"),
Address("", "", "", "443 Harbor View Rd", "Milford", "CT", "06460", address_residential_indicator="yes"),
Address("", "", "", "554 Steel City Pkwy", "Pittsburgh", "PA", "15222", address_residential_indicator="yes"),
Address("", "", "", "665 Market View St", "San Francisco", "CA", "94103", address_residential_indicator="yes"),
Address("", "", "", "776 Silicon Valley Blvd", "San Jose", "CA", "95112", address_residential_indicator="yes"),
Address("", "", "", "887 Sonoran Vista Rd", "Tucson", "AZ", "85701", address_residential_indicator="yes"),
Address("", "", "", "998 Atlantic Shore Dr", "Virginia Beach", "VA", "23451", address_residential_indicator="yes"),
Address("", "", "", "109 Sandia Crest Way", "Albuquerque", "NM", "87102", address_residential_indicator="yes"),
Address("", "", "", "210 Desert Willow Ln", "Gilbert", "AZ", "85233", address_residential_indicator="yes"),
Address("", "", "", "321 Lincoln Parkway", "Springfield", "IL", "62701", address_residential_indicator="yes"),
Address("", "", "", "432 Old Town Sq", "Alexandria", "VA", "22314", address_residential_indicator="yes"),
Address("", "", "", "543 Heart of America Blvd", "Kansas City", "MO", "64106", address_residential_indicator="yes"),
Address("", "", "", "654 Volunteer Way", "Knoxville", "TN", "37902", address_residential_indicator="yes"),
Address("", "", "", "765 Green Mountain Dr", "Lakewood", "CO", "80226", address_residential_indicator="yes"),
Address("", "", "", "876 Music Row Ave", "Nashville", "TN", "37203", address_residential_indicator="yes"),
Address("", "", "", "987 Gateway Arch Dr", "Saint Louis", "MO", "63101", address_residential_indicator="yes"),
Address("", "", "", "109 Puget Sound Ave", "Seattle", "WA", "98101", address_residential_indicator="yes"),
Address("", "", "", "101 W Abram St", "Arlington", "TX", "76010", address_residential_indicator="yes"),
Address("", "", "", "15151 E Alameda Pkwy", "Aurora", "CO", "80012", address_residential_indicator="yes"),
Address("", "", "", "150 N Capitol Blvd", "Boise", "ID", "83702", address_residential_indicator="yes"),
Address("", "", "", "200 E Washington St", "Indianapolis", "IN", "46204", address_residential_indicator="yes"),
Address("", "", "", "25 County Road 135", "Naples", "CO", "81137", address_residential_indicator="yes"),
Address("", "", "", "1 E 1st St", "Reno", "NV", "89501", address_residential_indicator="yes"),
Address("", "", "", "3900 Main St", "Riverside", "CA", "92522", address_residential_indicator="yes"),
Address("", "", "", "221 N 5th St", "Bismarck", "ND", "58501", address_residential_indicator="yes"),
Address("", "", "", "10210 Highway 521", "Fresno", "TX", "77545", address_residential_indicator="yes"),
Address("", "", "", "100 Old River Rd", "Lincoln", "RI", "02865", address_residential_indicator="yes"),
Address("", "", "", "200 N Spring St", "Los Angeles", "CA", "90012", address_residential_indicator="yes"),
Address("", "", "", "6 Park Row", "Mansfield", "MA", "02048", address_residential_indicator="yes"),
Address("", "", "", "200 E Wells St", "Milwaukee", "WI", "53202", address_residential_indicator="yes"),
Address("", "", "", "101 W Main St", "Madison", "IN", "47250", address_residential_indicator="yes"),
Address("", "", "", "57 E 1st St", "Mesa", "AZ", "85201", address_residential_indicator="yes"),
Address("", "", "", "300 N Main St", "Monroe", "NC", "28112", address_residential_indicator="yes"),
Address("", "", "", "1400 John F Kennedy Blvd", "Philadelphia", "PA", "19107", address_residential_indicator="yes"),
Address("", "", "", "10 Richmond Terrace", "Staten Island", "NY", "10301", address_residential_indicator="yes"),
Address("", "", "", "1350 Pennsylvania Ave NW", "Washington", "DC", "20004", address_residential_indicator="yes"),
Address("", "", "", "100 N Holliday St", "Baltimore", "MD", "21202", address_residential_indicator="yes"),
Address("", "", "", "1 Gary K Anderson Plaza", "Decatur", "IL", "62523", address_residential_indicator="yes"),
Address("", "", "", "101 City Hall Plaza", "Durham", "NC", "27701", address_residential_indicator="yes"),
Address("", "", "", "113 W Mountain St", "Fayetteville", "AR", "72701", address_residential_indicator="yes"),
Address("", "", "", "5850 W Glendale Ave", "Glendale", "AZ", "85301", address_residential_indicator="yes"),
Address("", "", "", "6890 GA-219", "Midland", "GA", "31820", address_residential_indicator="yes"),
Address("", "", "", "3939 N Drinkwater Blvd", "Scottsdale", "AZ", "85251", address_residential_indicator="yes"),
Address("", "", "", "12453 Highway 92", "Woodstock", "GA", "30188", address_residential_indicator="yes"),
Address("", "", "", "175 S Arizona Ave", "Chandler", "AZ", "85225", address_residential_indicator="yes"),
Address("", "", "", "801 Plum St", "Cincinnati", "OH", "45202", address_residential_indicator="yes"),
Address("", "", "", "317 N Jefferson Ave", "Covington", "LA", "70433", address_residential_indicator="yes"),
Address("", "", "", "2200 Second St", "Fort Myers", "FL", "33901", address_residential_indicator="yes"),
Address("", "", "", "300 W Washington St", "Greensboro", "NC", "27401", address_residential_indicator="yes"),
Address("", "", "", "1500 Industrial Dr", "Henderson", "TX", "75652", address_residential_indicator="yes"),
Address("", "", "", "3298 M-40", "Hamilton", "MI", "49419", address_residential_indicator="yes"),
Address("", "", "", "131 Market St", "Lewisburg", "PA", "17837", address_residential_indicator="yes"),
Address("", "", "", "2255 W Berry Ave", "Littleton", "CO", "80120", address_residential_indicator="yes"),
Address("", "", "", "100 Ann Edwards Ln", "Mount Pleasant", "SC", "29464", address_residential_indicator="yes"),
Address("", "", "", "8100 Ritchie Hwy", "Pasadena", "MD", "21122", address_residential_indicator="yes"),
Address("", "", "", "1520 K Ave", "Plano", "TX", "75074", address_residential_indicator="yes"),
Address("", "", "", "225 E Weatherspoon St", "Sanford", "NC", "27330", address_residential_indicator="yes"),
Address("", "", "", "609 Main St", "Wayne", "ME", "04284", address_residential_indicator="yes"),
Address("", "", "", "121 N Washington St", "Ashland", "PA", "17921", address_residential_indicator="yes"),
Address("", "", "", "300 N Pine St", "Burlington", "WI", "53105", address_residential_indicator="yes"),
Address("", "", "", "220 N 13th St", "Centerville", "IA", "52544", address_residential_indicator="yes"),
Address("", "", "", "111 East 2nd St", "Clayton", "NC", "27520", address_residential_indicator="yes"),
Address("", "", "", "505 S Vulcan Ave", "Encinitas", "CA", "92024", address_residential_indicator="yes"),
Address("", "", "", "300 Laporte Ave", "Fort Collins", "CO", "80521", address_residential_indicator="yes"),
Address("", "", "", "911 10th St", "Golden", "CO", "80401", address_residential_indicator="yes"),
Address("", "", "", "550 Central St", "Hudson", "NC", "28638", address_residential_indicator="yes"),
Address("", "", "", "901 Avenue C", "Katy", "TX", "77493", address_residential_indicator="yes"),
Address("", "", "", "104 N Main St", "Lancaster", "SC", "29720", address_residential_indicator="yes"),
Address("", "", "", "920 Broad St", "Newark", "NJ", "07102", address_residential_indicator="yes"),
Address("", "", "", "20120 E Mainstreet", "Parker", "CO", "80138", address_residential_indicator="yes"),
Address("", "", "", "26 Court St", "Plymouth", "MA", "02360", address_residential_indicator="yes"),
Address("", "", "", "10500 Civic Center Dr", "Rancho Cucamonga", "CA", "91730", address_residential_indicator="yes"),
Address("", "", "", "111 Maryland Ave", "Rockville", "MD", "20850", address_residential_indicator="yes"),
Address("", "", "", "915 I St", "Sacramento", "CA", "95814", address_residential_indicator="yes"),
Address("", "", "", "300 S Adams St", "Tallahassee", "FL", "32301", address_residential_indicator="yes"),
Address("", "", "", "984 Old Mill Run", "The Villages", "FL", "32162", address_residential_indicator="yes"),
Address("", "", "", "130 Penn St", "Westfield", "IN", "46074", address_residential_indicator="yes"),
Address("", "", "", "715 Mulberry St", "Waterloo", "IA", "50703", address_residential_indicator="yes"),
Address("", "", "", "401 E Gay St", "West Chester", "PA", "19380", address_residential_indicator="yes"),
Address("", "", "", "710 NW Wall St", "Bend", "OR", "97703", address_residential_indicator="yes"),
Address("", "", "", "501 S Buchanan St", "Amarillo", "TX", "79101", address_residential_indicator="yes"),
Address("", "", "", "104 Central St", "Auburn", "MA", "01501", address_residential_indicator="yes"),
Address("", "", "", "1501 Truxtun Ave", "Bakersfield", "CA", "93301", address_residential_indicator="yes"),
Address("", "", "", "201 W Palmetto Park Rd", "Boca Raton", "FL", "33432", address_residential_indicator="yes"),
Address("", "", "", "200 Burnett St", "Benton", "LA", "71006", address_residential_indicator="yes"),
Address("", "", "", "101 Old Main St", "Bradenton", "FL", "34205", address_residential_indicator="yes"),
Address("", "", "", "101 W 3rd St", "Dayton", "OH", "45402", address_residential_indicator="yes"),
Address("", "", "", "411 W 1st St", "Duluth", "MN", "55802", address_residential_indicator="yes"),
Address("", "", "", "1000 Englewood Pkwy", "Englewood", "CO", "80110", address_residential_indicator="yes"),
Address("", "", "", "1 E Main St", "Fort Wayne", "IN", "46802", address_residential_indicator="yes"),
Address("", "", "", "101 N Court St", "Frederick", "MD", "21701", address_residential_indicator="yes"),
Address("", "", "", "715 Princess Anne St", "Fredericksburg", "VA", "22401", address_residential_indicator="yes"),
Address("", "", "", "200 E University Ave", "Gainesville", "FL", "32601", address_residential_indicator="yes"),
Address("", "", "", "300 Monroe Ave NW", "Grand Rapids", "MI", "49503", address_residential_indicator="yes"),
Address("", "", "", "100 N Jefferson St", "Green Bay", "WI", "54301", address_residential_indicator="yes"),
Address("", "", "", "389 Spruce St", "Morgantown", "WV", "26505", address_residential_indicator="yes"),
Address("", "", "", "222 W Main St", "Pensacola", "FL", "32502", address_residential_indicator="yes"),
Address("", "", "", "6 Town Hall Dr", "Princeton", "MA", "01541", address_residential_indicator="yes"),
Address("", "", "", "8401 W Monroe St", "Peoria", "AZ", "85345", address_residential_indicator="yes"),
Address("", "", "", "15 Kellogg Blvd W", "Saint Paul", "MN", "55102", address_residential_indicator="yes"),
Address("", "", "", "2 E Bay St", "Savannah", "GA", "31401", address_residential_indicator="yes"),
Address("", "", "", "4000 Merrick Rd", "Seaford", "NY", "11783", address_residential_indicator="yes"),
Address("", "", "", "301 N Chestnut St", "Seymour", "IN", "47274", address_residential_indicator="yes"),
Address("", "", "", "808 W Spokane Falls Blvd", "Spokane", "WA", "99201", address_residential_indicator="yes"),
Address("", "", "", "301 Walnut St", "Windsor", "CO", "80550", address_residential_indicator="yes"),
Address("", "", "", "101 N Main St", "Winston-Salem", "NC", "27101", address_residential_indicator="yes"),
Address("", "", "", "455 Main St", "Worcester", "MA", "01608", address_residential_indicator="yes"),
Address("", "", "", "101 S George St", "York", "PA", "17401", address_residential_indicator="yes"),
Address("", "", "", "5501 Blue Hole Rd", "Antioch", "TN", "37013", address_residential_indicator="yes"),
Address("", "", "", "301 College Ave", "Athens", "GA", "30601", address_residential_indicator="yes"),
Address("", "", "", "400 S Vicentia Ave", "Corona", "CA", "92882", address_residential_indicator="yes"),
Address("", "", "", "1000 Webster St", "Fairfield", "CA", "94533", address_residential_indicator="yes"),
Address("", "", "", "19330 Frederick Rd", "Germantown", "MD", "20876", address_residential_indicator="yes"),
Address("", "", "", "150 E Main St", "Hillsboro", "OR", "97123", address_residential_indicator="yes"),
Address("", "", "", "5770 Rockfish Rd", "Hope Mills", "NC", "28348", address_residential_indicator="yes"),
Address("", "", "", "160 6th Ave E", "Hendersonville", "NC", "28792", address_residential_indicator="yes"),
Address("", "", "", "160 S Memphis St", "Holly Springs", "MS", "38635", address_residential_indicator="yes"),
Address("", "", "", "491 E Pioneer Ave", "Homer", "AK", "99603", address_residential_indicator="yes"),
Address("", "", "", "8103 Sandy Spring Rd", "Laurel", "MD", "20707", address_residential_indicator="yes"),
Address("", "", "", "411 W Ocean Blvd", "Long Beach", "CA", "90802", address_residential_indicator="yes"),
Address("", "", "", "1001 W Center St", "Manteca", "CA", "95337", address_residential_indicator="yes"),
Address("", "", "", "400 W Broadway Ave", "Maryville", "TN", "37801", address_residential_indicator="yes"),
Address("", "", "", "116 N Cherry St", "Magnolia", "MS", "39652", address_residential_indicator="yes"),
Address("", "", "", "125 N Main St", "Memphis", "TN", "38103", address_residential_indicator="yes"),
Address("", "", "", "525 Canton Ave", "Milton", "MA", "02186", address_residential_indicator="yes"),
Address("", "", "", "101 Old Plantersville Rd", "Montgomery", "TX", "77356", address_residential_indicator="yes"),
Address("", "", "", "937 Broadway St", "Myrtle Beach", "SC", "29577", address_residential_indicator="yes"),
Address("", "", "", "8706 Moores Mill Rd", "New Market", "AL", "35761", address_residential_indicator="yes"),
Address("", "", "", "110 SE Watula Ave", "Ocala", "FL", "34471", address_residential_indicator="yes"),
Address("", "", "", "100 E Santa Fe St", "Olathe", "KS", "66061", address_residential_indicator="yes"),
Address("", "", "", "22 W Burdick St", "Oxford", "MI", "48371", address_residential_indicator="yes"),
Address("", "", "", "10500 N Military Trl", "Palm Beach Gardens", "FL", "33410", address_residential_indicator="yes"),
Address("", "", "", "311 Vernon St", "Roseville", "CA", "95678", address_residential_indicator="yes"),
Address("", "", "", "38 Hill St", "Roswell", "GA", "30075", address_residential_indicator="yes"),
Address("", "", "", "385 S Goliad St", "Rockwall", "TX", "75087", address_residential_indicator="yes"),
Address("", "", "", "100 Main St", "Spring", "TX", "77373", address_residential_indicator="yes"),
Address("", "", "", "5 Beach Rd", "Salisbury", "MA", "01952", address_residential_indicator="yes"),
Address("", "", "", "100 Santa Rosa Ave", "Santa Rosa", "CA", "95404", address_residential_indicator="yes"),
Address("", "", "", "1565 1st St", "Sarasota", "FL", "34236", address_residential_indicator="yes"),
Address("", "", "", "415 W 6th St", "Vancouver", "WA", "98660", address_residential_indicator="yes"),
Address("", "", "", "1838 Emerald Hill Ln", "Westminster", "MD", "21157", address_residential_indicator="yes"),
Address("", "", "", "200 E Main St", "Canton", "GA", "30114", address_residential_indicator="yes"),
Address("", "", "", "701 E Broadway", "Columbia", "MO", "65201", address_residential_indicator="yes"),
Address("", "", "", "735 8th St S", "Naples", "FL", "34102", address_residential_indicator="yes"),
Address("", "", "", "2600 Fresno St", "Fresno", "CA", "93721", address_residential_indicator="yes"),
Address("", "", "", "118 Church Ave SW", "Cleveland", "TN", "37311", address_residential_indicator="yes"),
Address("", "", "", "240 Water St", "Henderson", "NV", "89015", address_residential_indicator="yes"),
Address("", "", "", "555 S 10th St", "Lincoln", "NE", "68508", address_residential_indicator="yes"),
Address("", "", "", "800 N Front St", "Wilmington", "NC", "28401", address_residential_indicator="yes"),
"""

ADDRESS_PATTERN = re.compile(
    r'Address\([^)]*?"[^"]*",\s*"[^"]*",\s*"[^"]*",\s*"[^"]*",\s*"([^"]+)",\s*"([^"]+)"',
    re.IGNORECASE,
)


def parse_city_state(lines):
    city_states = set()
    for line in lines:
        match = ADDRESS_PATTERN.search(line)
        if not match:
            continue
        city = match.group(1).strip()
        state = match.group(2).strip()
        if city and state:
            city_states.add((city.casefold(), state.casefold()))
    return city_states


def main():
    city_states = parse_city_state(ADDRESSES_RAW.splitlines())
    if not city_states:
        raise SystemExit("No valid Address(...) rows found in addresses.txt.")

    with open(INPUT_PAGE3_PATH, "r", newline="", encoding="utf-8") as infile:
        reader = csv.DictReader(infile)
        fieldnames = reader.fieldnames or []
        if "ToCity" not in fieldnames or "ToState" not in fieldnames:
            raise ValueError("page3.csv must include ToCity and ToState columns.")

        rows = list(reader)

        with open(OUTPUT_PAGE4_PATH, "w", newline="", encoding="utf-8") as outfile:
            writer = csv.DictWriter(outfile, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                city = str(row.get("ToCity", "")).strip().casefold()
                state = str(row.get("ToState", "")).strip().casefold()
                if (city, state) in city_states:
                    writer.writerow(row)

    filtered_rows = [
        row for row in rows
        if (str(row.get("ToCity", "")).strip().casefold(), str(row.get("ToState", "")).strip().casefold()) in city_states
    ]
    if not filtered_rows:
        print("No rows matched the address list; skipping unmatched destination report.")
        return

    all_destinations = sorted({f"{row['ToCity']}, {row['ToState']}" for row in rows})
    allowed_destinations = {
        f"{city.title()}, {state.upper()}" for city, state in city_states
    }
    not_in_list = [dest for dest in all_destinations if dest not in allowed_destinations]

    print("Destinations in page3.csv not in the provided list:")
    if not_in_list:
        for dest in not_in_list:
            print(f"- {dest}")
    else:
        print("- none")


if __name__ == "__main__":
    main()
