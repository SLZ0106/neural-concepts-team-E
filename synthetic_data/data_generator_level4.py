import json
import random

# We keep the same schema as your old dataset:
# {id, statement, counts, mean, variance, domain}

DOMAINS = ["weather", "finance", "health", "sports", "education", "operations"]

# Avoid obvious lexical shortcuts (optional but recommended)
FORBIDDEN = [
    "uncertain", "uncertainty", "random", "probab", "likely", "maybe", "might", "could",
    "chance", "risk", "risky", "odds", "guarantee", "guaranteed", "variance", "entropy"
]

def has_digit(s: str) -> bool:
    return any(ch.isdigit() for ch in s)

def contains_forbidden(s: str) -> bool:
    low = s.lower()
    return any(w in low for w in FORBIDDEN)

# --- Event pools (no digits; no explicit uncertainty words) ---
EVENTS = {
    "low": {
        "weather": [
            "The temperature inside a climate controlled lab stays stable during the day.",
            "An indoor greenhouse keeps humidity steady using automated controls.",
            "A sealed room maintains constant airflow set by the ventilation system.",
        ],
        "finance": [
            "A contract specifies a fixed payment schedule and the funds are already placed in escrow.",
            "A subscription renews automatically under an existing agreement with no changes to terms.",
            "A bond coupon payment follows a published schedule under standard settlement rules.",
        ],
        "health": [
            "A calibrated lab instrument repeats the same assay under the same protocol.",
            "A medication dose is delivered by a verified infusion pump with fixed settings.",
            "A sample is processed in a controlled environment following a standard procedure.",
        ],
        "sports": [
            "A team rehearses a set play in practice with no defenders and repeats the same sequence.",
            "A player performs a drill alone using the same routine and timing each attempt.",
            "A coach runs a predetermined training circuit with fixed steps and repetitions.",
        ],
        "education": [
            "A student recites a memorized passage practiced many times in the same order.",
            "A spelling quiz uses a fixed answer key and straightforward grading rules.",
            "A worksheet problem has a single correct answer and the solution method is standard.",
        ],
        "operations": [
            "A nightly backup runs on a dedicated server with stable load and fixed parameters.",
            "A production line operates with fixed inputs and automated quality checks.",
            "A scheduled report is generated from a frozen dataset using the same pipeline.",
        ],
    },
    "medium": {
        # "medium" = well-defined randomness / mechanical randomness (dice, coin, draw)
        "weather": [
            "A leaf released into a swirling breeze lands in a specific spot on the ground.",
            "A dust particle released into a turbulent airflow settles somewhere on a surface.",
        ],
        "finance": [
            "A sealed envelope is drawn from a thoroughly mixed stack and opened.",
            "A raffle ticket is selected from a container after mixing.",
        ],
        "health": [
            "A sample tube is selected from a tray after the positions are mixed.",
            "A swab kit is drawn from a box after the kits are shuffled.",
        ],
        "sports": [
            "A coin toss decides which team starts with possession.",
            "A shuffled deck reveals a suit when the top card is turned over.",
        ],
        "education": [
            "A student guesses an answer on a multiple choice question without reading it.",
            "A classroom draws one name from a thoroughly mixed hat of name cards.",
        ],
        "operations": [
            "A coin flip determines which queue receives the next request.",
            "A die roll chooses among several prewritten options for the next step.",
        ],
    },
    "high": {
        # "high" = open-world, many hidden variables (earnings-call-like)
        "weather": [
            "It rains tomorrow in a coastal city where wind and cloud patterns shift quickly.",
            "A storm forms near the region and the local forecast depends on moving fronts.",
            "A commute faces changing road conditions due to scattered showers across the area.",
        ],
        "finance": [
            "A company exceeds its quarterly expectations because demand shifts and orders arrive late.",
            "A new product launch succeeds in the market as competitors and customer preferences react.",
            "Revenue next quarter changes as renewals, churn, and new deals evolve through the period.",
        ],
        "health": [
            "A patient responds to treatment as sleep, stress, adherence, and background conditions vary.",
            "Recovery time changes with daily routines, diet, and differences in individual physiology.",
            "Symptoms improve or worsen as multiple interacting factors shift over time.",
        ],
        "sports": [
            "A team wins an away match as strategy adjusts and both sides react in real time.",
            "A player performance changes under pressure with crowd noise and changing defense.",
            "The outcome of a close game shifts as momentum and decisions evolve throughout play.",
        ],
        "education": [
            "A student performs in an open ended discussion where questions follow the flow of debate.",
            "A panel evaluates a presentation where preferences differ and interpretation affects scoring.",
            "A group project outcome changes as coordination, motivation, and feedback evolve over time.",
        ],
        "operations": [
            "A supply chain delivery arrives on time as shipping delays and inspections vary across routes.",
            "System latency changes as external traffic bursts and user demand fluctuates through the day.",
            "A service incident resolves quickly depending on diagnosis paths and cascading dependencies.",
        ],
    }
}
# -----------------------------
# Expand pools (add more statements)
# -----------------------------

EVENTS["low"]["weather"].extend([
    "A thermostat keeps the office temperature steady through the day.",
    "A closed incubator maintains a stable environment while samples are stored.",
    "A building automation system holds airflow at a constant setting indoors.",
    "Inside a server room, cooling systems regulate heat output to a steady range.",
    "In a sealed container, moisture levels remain stable under controlled conditions.",
    "An indoor pool area uses dehumidifiers to keep humidity consistent.",
    "A laboratory hood runs at a fixed fan setting during routine operation.",
    "An insulated chamber keeps temperature nearly constant for the duration of the test.",
    "A controlled enclosure keeps light and heat stable for plant growth.",
    "A climate controlled warehouse maintains a steady temperature for storage.",
    "The indoor air handling system runs on a fixed schedule with constant output.",
    "A sealed test box keeps airflow isolated from outdoor conditions.",
])

EVENTS["low"]["finance"].extend([
    "A fixed fee service charges the same amount each billing cycle under the agreement.",
    "An invoice is paid automatically under standing instructions with no changes.",
    "A supplier contract locks the unit price for the delivery period.",
    "A payroll deposit follows a standard schedule for salaried employees.",
    "A lease payment is scheduled under a signed contract with agreed terms.",
    "A subscription plan renews under the same plan rules and billing cadence.",
    "A savings account interest credit follows the bank’s posted schedule.",
    "A vendor maintenance plan is prepaid and delivered under a fixed scope.",
    "A government fee is assessed under a published rule with a fixed rate.",
    "A service retainer is billed under a fixed agreement already in place.",
    "A utility autopay processes the same base charge each cycle under the plan.",
    "A booked transfer is executed under standard settlement once authorized.",
])

EVENTS["low"]["health"].extend([
    "A thermometer reads the same reference source under repeated checks.",
    "A lab centrifuge runs the same protocol with identical settings each run.",
    "A standard screening questionnaire is scored by a fixed key.",
    "A pharmacy dispenses a medication using a verified prescription and workflow.",
    "A controlled sample is weighed on a calibrated scale under the same procedure.",
    "A clinical device runs a self test that returns the same status under normal conditions.",
    "A routine blood pressure measurement is taken after resting in a quiet room.",
    "A sterile preparation follows a fixed checklist in a controlled environment.",
    "A specimen is stored in a temperature controlled unit with stable settings.",
    "A lab culture grows under constant incubation settings for the duration.",
    "A dosage is prepared by a standardized protocol with verified steps.",
    "A medical imaging calibration phantom produces consistent readings each time.",
])

EVENTS["low"]["sports"].extend([
    "A player repeats a warm up routine alone following the same sequence.",
    "A coach runs a drill with fixed cues and a predetermined order of actions.",
    "A team practices passing patterns without defenders using a set script.",
    "A sprinter performs starts on a track under the same coaching instructions.",
    "A swimmer follows a fixed lap set in training with consistent pacing cues.",
    "A gymnast rehearses a routine on a mat in a quiet practice session.",
    "A player shoots from a marked spot in an empty gym following the same form.",
    "A team runs conditioning laps with a fixed route and fixed cadence guidance.",
    "A goalkeeper practices controlled catches from a coach at a steady rhythm.",
    "A batter practices tee hits with the ball placed in the same position each time.",
    "A basketball player repeats free throw form drills alone in practice.",
    "A team rehearses a kickoff routine with no opposing pressure during practice.",
])

EVENTS["low"]["education"].extend([
    "A student copies a worked example following the same steps as shown.",
    "A short quiz uses a fixed answer key and automatic scoring rules.",
    "A student reads a scripted speech they rehearsed in the same order.",
    "A worksheet uses direct lookup facts from a provided reference sheet.",
    "A grading assistant applies a rubric with exact matches for correct responses.",
    "A spelling list test follows a standard key with straightforward checking.",
    "A student practices a memorized definition and repeats it verbatim.",
    "A fill in the blank exercise matches a fixed set of expected words.",
    "A closed book recall task asks for a memorized list in a fixed order.",
    "A practice exam uses known solutions and deterministic scoring.",
    "A student repeats the same flashcard deck with the same prompts each time.",
    "A tutorial follows a step by step script with predefined outputs.",
])

EVENTS["low"]["operations"].extend([
    "A batch job runs on a dedicated machine with stable inputs and fixed configuration.",
    "A scheduled maintenance task follows a checklist with the same steps each time.",
    "A routine data export runs from a fixed snapshot using the same query.",
    "A service health check runs a fixed set of endpoints with the same timeout settings.",
    "A controlled deployment is performed on a staging system with identical configuration.",
    "A periodic report pulls from a frozen table with a stable schema.",
    "A continuous integration pipeline runs the same tests under the same environment.",
    "A cron task processes a fixed folder with a stable naming convention.",
    "A backup verification checks a known file set using the same method each night.",
    "A monitoring alert rule evaluates the same metric under a constant threshold rule.",
    "A container image build uses a fixed base and the same build steps.",
    "A static website build runs from a pinned commit with stable dependencies.",
])

# ----- MEDIUM (mechanical randomness / structured chance, no probability words) -----

EVENTS["medium"]["weather"].extend([
    "A raindrop released from a leaf falls and lands at a particular point on the ground.",
    "A small paper strip tossed into a swirling draft lands facing one direction.",
    "A puff of smoke disperses in a room and its final shape varies from trial to trial.",
    "A bubble released into moving air drifts and settles at a particular location.",
    "A feather released in a light indoor draft lands in a different spot each time.",
    "A tiny bead dropped into flowing water settles into a particular corner.",
    "A speck of dust floats in air currents and ends up on one surface among many.",
    "A lightweight ribbon flutters in a breeze and settles in one orientation.",
    "A pollen grain released into air ends up on one of many nearby surfaces.",
    "A leaf twirled and released lands edge up or flat on the ground.",
    "A droplet rolls on a slightly tilted surface and stops at one point.",
    "A small scrap of paper tossed gently lands on one of several tiles.",
])

EVENTS["medium"]["finance"].extend([
    "A shuffled stack of sealed envelopes is mixed and one envelope is opened.",
    "A card is drawn from a well shuffled deck and its suit is observed.",
    "A token is drawn from a bag after mixing the tokens thoroughly.",
    "A ticket is selected from a container after the tickets are mixed.",
    "A roulette wheel is spun and the landing pocket is observed.",
    "A spinner is flicked and the pointer stops at one segment.",
    "A jar of mixed slips is shaken and one slip is pulled without looking.",
    "A dice cup is shaken and the top face of the die is observed after the roll.",
    "A shuffled deck is cut and the top card is revealed.",
    "A handful of coins is shaken in a box and one coin is selected.",
    "A receipt is drawn from a mixed pile for a spot check.",
    "A lottery ball machine releases one ball and the marking is read.",
])

EVENTS["medium"]["health"].extend([
    "A sealed bag of labeled swabs is mixed and one swab is selected.",
    "A tray of sample tubes is shuffled and one tube is picked without looking.",
    "A box of identical kits is mixed and one kit is opened for inspection.",
    "A lab chooses one specimen from a mixed batch for a quality check.",
    "A pill bottle is shaken and one tablet is selected from identical tablets.",
    "A set of patient charts is mixed and one chart is selected for audit.",
    "A stack of appointment slips is mixed and one slip is selected.",
    "A collection of vials is rotated and one vial is drawn for testing.",
    "A set of test strips is mixed and one strip is selected for a run.",
    "A row of identical containers is shuffled and one is opened.",
    "A group of identical syringes is mixed and one is selected for verification.",
    "A batch of labels is mixed and one label is selected for checking.",
])

EVENTS["medium"]["sports"].extend([
    "A coin is flipped to decide which team starts with possession.",
    "A referee draws one card from a shuffled set to choose the starting side.",
    "A die is rolled to select among several practice drills.",
    "A spinner is used to choose the next training station for the team.",
    "A coach draws one slip from a hat to pick the next shooter.",
    "A deck of cards is shuffled and one card is drawn to choose the next drill.",
    "A set of numbered balls is mixed and one ball is drawn to pick a team.",
    "A tournament bracket position is chosen by drawing one token from a bag.",
    "A jersey is drawn to determine the next matchup order.",
    "A puck is dropped and bounces before settling on one face of a marked disk.",
    "A coin toss decides the first serve in a friendly match.",
    "A shuffled stack of cue cards is used to pick the next exercise.",
])

EVENTS["medium"]["education"].extend([
    "A student guesses on a multiple choice item without reading it.",
    "A teacher shuffles name cards and draws one card to call on a student.",
    "A class draws one slip from a mixed bowl to choose the next topic.",
    "A stack of flashcards is shuffled and the next card is revealed.",
    "A quiz question is selected by drawing one card from a shuffled deck.",
    "A peer review assignment is made by drawing slips from a hat.",
    "A student picks one book from a mixed stack without looking.",
    "A group forms pairs by drawing colored cards from a shuffled pack.",
    "A classroom chooses the next presenter by drawing a folded slip.",
    "A worksheet version is assigned by drawing one card from mixed versions.",
    "A student picks one of several prompts by drawing a sealed envelope.",
    "A discussion order is set by drawing names from a mixed pile.",
])

EVENTS["medium"]["operations"].extend([
    "A request is assigned to one of several workers by drawing a token from a bag.",
    "A load balancer selects a backend from a shuffled list for a demonstration.",
    "A cache eviction chooses one entry from a mixed set under a simple rule.",
    "A scheduler picks one job from a mixed queue for the next run.",
    "A support rotation chooses the next on call person by drawing a slip.",
    "A log line is selected from a mixed sample for inspection.",
    "A test harness selects one configuration by drawing from a shuffled set.",
    "A deployment chooses one canary host from a mixed pool for the first rollout step.",
    "A quality check picks one item from a mixed batch for inspection.",
    "A file is chosen from a mixed folder listing for a spot check.",
    "A routing demo picks one path from a shuffled list to simulate a hop.",
    "A bucket is selected by drawing one label from a shuffled set.",
])

# ----- HIGH (open-world, many hidden factors; earnings-call-like) -----

EVENTS["high"]["weather"].extend([
    "A city experiences scattered showers as local conditions shift through the day.",
    "A flight faces delays due to changing wind patterns and evolving conditions at the airport.",
    "Fog forms near the coast and visibility changes as conditions move inland.",
    "A winter mix affects roads and conditions vary across neighborhoods.",
    "A heat wave intensity shifts as cloud cover changes and winds move in.",
    "A thunderstorm impacts the area as conditions develop unevenly across regions.",
    "Outdoor humidity changes as air masses move and local terrain affects airflow.",
    "A morning commute is affected by patchy ice as temperatures vary by location.",
    "A beach day comfort changes as breezes strengthen and clouds move overhead.",
    "An outdoor event faces changing conditions as weather cells pass nearby.",
    "Visibility changes along a highway as low clouds drift across the route.",
    "A river level shifts as upstream rainfall varies across the watershed.",
])

EVENTS["high"]["finance"].extend([
    "A company’s quarterly results depend on late customer decisions and shifting demand.",
    "A merger outcome changes as regulatory review, negotiation, and market reaction evolve.",
    "A subscription business changes performance as churn and retention shift with competition.",
    "A retailer’s season performance varies as consumer preferences and promotions interact.",
    "A software rollout affects revenue as adoption patterns differ across customer segments.",
    "A supplier price changes as raw material costs and logistics disruptions evolve.",
    "A new market entry outcome changes as local competition responds and demand shifts.",
    "A forecast changes as inventory constraints and shipping delays interact with demand.",
    "A product margin changes as mix shifts and discounts are adjusted throughout the quarter.",
    "A company’s guidance changes as pipeline conversion and renewals move during the period.",
    "A growth target outcome changes as hiring, ramp time, and execution differ across teams.",
    "A services business varies as utilization and project timing shift across clients.",
])

EVENTS["high"]["health"].extend([
    "A patient’s recovery course changes as adherence, sleep, stress, and support vary over time.",
    "A treatment outcome differs as immune response and concurrent conditions interact.",
    "Symptoms fluctuate as lifestyle, exposure, and routines change through the week.",
    "A therapy response varies as motivation, side effects, and follow up behavior evolve.",
    "A diagnosis timeline changes as symptoms shift and access to care varies.",
    "A rehabilitation outcome varies as effort, pain, and daily activity levels change.",
    "A medication response differs as metabolism and interactions vary across individuals.",
    "A patient’s sleep quality changes as schedule, stress, and environment evolve.",
    "A chronic condition trajectory shifts as routines and triggers change over time.",
    "A hospital stay length varies as complications and care coordination differ.",
    "An infection course changes as exposure, immunity, and behavior interact.",
    "A mental health outcome shifts as support, routines, and stressors evolve.",
])

EVENTS["high"]["sports"].extend([
    "A close match outcome changes as tactics adjust and players react to momentum shifts.",
    "A championship series outcome varies as injuries, form, and adjustments evolve game by game.",
    "A team performance shifts as fatigue and in game decisions interact.",
    "A player scoring output changes as defensive schemes adapt throughout the match.",
    "A comeback attempt depends on turnovers, pace changes, and coaching choices.",
    "A rivalry game outcome shifts as emotions and decisions evolve under pressure.",
    "A tournament run changes as matchups and preparation vary across rounds.",
    "A late game outcome depends on refereeing decisions and execution under stress.",
    "A team’s away performance changes as travel, crowd, and rhythm interact.",
    "A goalkeeper performance changes as shots vary and pressure builds.",
    "A team’s coordination changes as substitutions and strategy shifts occur.",
    "A final score changes as tempo and bold play evolve late in the game.",
])

EVENTS["high"]["education"].extend([
    "A student’s performance on an open response exam varies with interpretation and reasoning path.",
    "A group project outcome changes as coordination, motivation, and division of labor evolve.",
    "A class discussion quality varies as participation patterns and topics shift.",
    "A presentation score changes as judges weigh content and delivery differently.",
    "A peer review outcome varies as reviewers focus on different strengths and weaknesses.",
    "A learning outcome changes as study habits and feedback cycles differ across weeks.",
    "An oral exam result varies as questions adapt to the student’s responses.",
    "A course grade changes as late submissions, revisions, and feedback loops interact.",
    "A thesis progress varies as research obstacles and iteration cycles evolve.",
    "A mentorship effect changes as communication and expectations evolve over time.",
    "A student engagement changes as workload, interest, and support vary during the term.",
    "A reading comprehension outcome varies as background knowledge and fatigue interact.",
])

EVENTS["high"]["operations"].extend([
    "An incident resolution time varies as diagnosis paths and dependencies unfold during response.",
    "System performance changes as traffic patterns shift and shared resources contend.",
    "A release outcome varies as integration issues and environment differences appear during rollout.",
    "A delivery timeline changes as suppliers, shipping, and inspections interact across the route.",
    "A customer support backlog changes as demand spikes and staffing availability varies.",
    "A service reliability outcome changes as cascading failures and recovery steps interact.",
    "A migration completion time varies as unexpected edge cases and retries occur.",
    "A production issue impact changes as user behavior and system load evolve.",
    "A capacity planning outcome changes as adoption, churn, and usage patterns shift.",
    "A vendor delivery varies as upstream constraints and schedule changes propagate.",
    "A scaling effort result varies as bottlenecks shift and tuning choices interact.",
    "A rollout pace changes as monitoring signals and mitigation steps evolve.",
])

# Map semantic uncertainty -> numeric "variance" label (0~400)
# Keep it simple: three bands with small jitter so it isn't trivially discretized.
VAR_BANDS = {
    "low":   (0.0, 80.0),
    "medium":(140.0, 260.0),
    "high":  (320.0, 400.0),
}

def generate_event_dataset(N=600, seed=7, out_path="event_uncertainty_600.json"):
    rng = random.Random(seed)
    data = []
    mean = 10.0

    # counts kept constant to avoid becoming a shortcut (still matches your schema)
    counts = [20, 20, 20]

    levels = ["low", "medium", "high"]
    # balanced levels
    level_cycle = (levels * ((N // 3) + 1))[:N]
    rng.shuffle(level_cycle)

    for i in range(1, N + 1):
        level = level_cycle[i-1]
        domain = DOMAINS[(i - 1) % len(DOMAINS)]
        pool = EVENTS[level][domain]
        statement = rng.choice(pool)

        # sanity: avoid digits and forbidden cue words
        if has_digit(statement):
            raise RuntimeError(f"Digit found in statement: {statement}")
        if contains_forbidden(statement):
            raise RuntimeError(f"Forbidden cue found in statement: {statement}")

        vmin, vmax = VAR_BANDS[level]
        variance = rng.uniform(vmin, vmax)

        item = {
            "id": i,
            "statement": statement,
            "counts": counts,
            "mean": mean,
            "variance": round(float(variance), 6),
            "domain": domain
        }
        data.append(item)

    with open(out_path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Wrote {out_path} with {len(data)} items")
    print("First 3 items:\n", json.dumps(data[:3], indent=2))

if __name__ == "__main__":
    generate_event_dataset()
