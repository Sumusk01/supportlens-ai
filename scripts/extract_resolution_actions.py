import re
import pandas as pd
from pathlib import Path


# ============================================================
# Configuration
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_FILE = BASE_DIR / "data" / "historical_replies_sample_100.csv"
OUTPUT_FILE = BASE_DIR / "data" / "historical_replies_actions_100.csv"


# ============================================================
# Action taxonomy
# ============================================================

ACTIONS = [
    "UPDATE_SOFTWARE",
    "ANNOUNCE_FIX_UPDATE",
    "RESTART_DEVICE",
    "CHECK_SETTINGS",
    "CHECK_CONNECTION",
    "CHECK_ACCOUNT",
    "CHECK_APP_STORE",
    "CHECK_BATTERY",
    "PROVIDE_TROUBLESHOOTING",
    "PROVIDE_WORKAROUND",
    "PROVIDE_INFORMATION",
    "PROVIDE_ARTICLE",
    "ASK_DIAGNOSTIC_QUESTION",
    "ASK_FOR_DETAILS",
    "CONTACT_SUPPORT_DM",
    "ROUTE_TO_SPECIALIST",
    "ROUTE_FEEDBACK",
    "ROUTE_LANGUAGE_SUPPORT",
    "NO_CLEAR_ACTION",
]


# ============================================================
# Text utilities
# ============================================================

def normalize_text(text):
    """Normalize text for rule matching."""

    if pd.isna(text):
        return ""

    text = str(text)

    text = text.replace("&gt;", ">")
    text = text.replace("&lt;", "<")
    text = text.replace("&amp;", "&")

    text = re.sub(r"\s+", " ", text).strip()

    return text


def contains_any(text, patterns):
    """Return True if any regex pattern matches."""

    for pattern in patterns:
        if re.search(pattern, text, flags=re.IGNORECASE):
            return True

    return False


def extract_urls(text):
    """Extract URLs from a reply."""

    return re.findall(r"https?://\S+", text)


# ============================================================
# URL handling
# ============================================================

DM_LINK_PATTERNS = [
    r"gdrqu22ypt",
    r"gdrqu22yp",
    r"gdrqu2kzhr",
]


def is_dm_url(url):
    """Return True if URL is a known AppleSupport DM link."""

    return any(
        re.search(pattern, url, flags=re.IGNORECASE)
        for pattern in DM_LINK_PATTERNS
    )


def has_resource_link(text):
    """
    Return True when the reply contains a non-DM URL.
    """

    urls = extract_urls(text)

    for url in urls:
        if not is_dm_url(url):
            return True

    return False


# ============================================================
# Resolution action detectors
# ============================================================

def detect_announce_fix_update(text):
    """
    Detect communication that a software update contains
    or addresses a fix.
    """

    patterns = [
        r"\breleased iOS\b",
        r"\breleased an? iOS update\b",
        r"\brecently released\b.*\b(?:iOS|update)\b",
        r"\ban update has been released\b",
        r"\bcontains a fix\b",
        r"\bcontains fixes\b",
        r"\baddresses this issue\b",
        r"\baddresses the issue\b",
        r"\bfixed in\b.*\b(?:update|iOS|software)\b",
        r"\bresolved in\b.*\b(?:update|iOS|software)\b",
        r"\bfuture software update\b",
    ]

    return contains_any(text, patterns)


def detect_update_software(text):
    """
    Detect an actual recommendation/instruction to install
    or proceed with a software update.
    """

    patterns = [
        r"\bplease update\b",
        r"\brecommend updating\b",
        r"\brecommend you update\b",
        r"\bproceed with the update\b",
        r"\bupdate to iOS\b",
        r"\bupdate your (?:device|iPhone|iPad|software)\b",
        r"\bupdate .* to iOS\b",
        r"\bback up\b.*\bprior to updating\b",
        r"\bprior to updating\b",
    ]

    return contains_any(text, patterns)


def detect_restart(text):
    """
    Detect restart/reboot actions or restart-related questions.

    In the labeled support-reply examples, asking whether the
    customer has restarted is treated as a restart-related action.
    """

    patterns = [
        # Explicit restart instructions
        r"\bforce restart\b",
        r"\btry to force a restart\b",
        r"\btry restarting\b",
        r"\btry to restart\b",
        r"\brestart your\b",
        r"\brestart the\b",
        r"\brestart your device\b",
        r"\brestart the device\b",

        # Restart-related diagnostic questions
        r"\bhave you restarted\b",
        r"\bhave you tried restarting\b",
        r"\bhave you had a chance to restart\b",
        r"\bdid you restart\b",
        r"\bhave you restarted normally\b",

        # Reboot
        r"\breboot your\b",
        r"\breboot the\b",
        r"\bhave you rebooted\b",
        r"\bdid you reboot\b",

        # Power-cycle instructions
        r"\bturn (?:it|the device|your device) off\b",
        r"\bturn (?:it|the device|your device) back on\b",
    ]

    return contains_any(text, patterns)


def detect_settings(text):
    """
    Detect actual settings troubleshooting.

    Merely saying:
    'Settings > General > About'
    is NOT enough because that may simply tell the
    customer where to find their iOS version.
    """

    patterns = [
        r"\bsettings\b.*\b(?:enable|disable|turn|change|adjust)\b",
        r"\b(?:enable|disable|turn on|turn off)\b.*\bsettings\b",
        r"\bsettings\b.*\btoggle\b",
        r"\btoggle\b.*\bsetting\b",
    ]

    return contains_any(text, patterns)


def detect_connection(text):
    """Detect concrete connection/network checking actions."""

    patterns = [
        # Explicitly checking network/connection state
        r"\bcheck\b.*\bwi[- ]?fi\b",
        r"\bcheck\b.*\bnetwork\b",
        r"\bcheck\b.*\bconnection\b",

        # Explicit connection setup/action
        r"\bconnect\b.*\bwi[- ]?fi\b",
        r"\bconnect to\b.*\bwi[- ]?fi\b",

        # Cellular/network settings
        r"\bcellular\b.*\bsettings\b",
        r"\bcarrier\b.*\bcheck\b",
    ]

    return contains_any(text, patterns)


def detect_account(text):
    """Detect concrete Apple ID/account troubleshooting."""

    patterns = [
        r"\bcheck\b.*\bapple id\b",
        r"\bapple id\b.*\bpassword\b",
        r"\breset\b.*\bpassword\b",
        r"\bsign in\b.*\baccount\b",
        r"\blog in\b.*\baccount\b",
        r"\baccount\b.*\bsign in\b",
    ]

    return contains_any(text, patterns)


def detect_app_store(text):
    """Detect App Store/download/purchase troubleshooting."""

    patterns = [
        r"\bapp store\b.*\b(?:download|purchase|billing)\b",
        r"\bdownload\b.*\bapp\b",
        r"\bapp\b.*\bdownload\b",
        r"\bpurchase\b.*\bapp store\b",
        r"\bbilling\b.*\bapp store\b",
    ]

    return contains_any(text, patterns)


def detect_battery(text):
    """Detect concrete battery troubleshooting."""

    # Battery Life appearing only as the title/topic of an
    # article should not be interpreted as a battery action.
    if re.search(r"\barticle\b.*\bbattery life\b", text, flags=re.IGNORECASE):
        return False

    patterns = [
        r"\bcheck\b.*\bbattery\b",
        r"\bbattery\b.*\bsettings\b",
        r"\bmaximize battery life\b",
        r"\bbattery life\b.*\b(?:guide|article|steps)\b",
        r"\bcharging\b.*\bsteps\b",
        r"\bcharge\b.*\btroubleshoot\b",
    ]

    return contains_any(text, patterns)


def detect_workaround(text):
    """Detect workaround language."""

    patterns = [
        r"\bworkaround\b",
        r"\bwork around\b",
        r"\buntil it(?:'s| is) fixed\b",
        r"\buntil .* is fixed\b",
        r"\bin the meantime\b",
        r"\btemporary solution\b",
        r"\btemporary fix\b",
    ]

    return contains_any(text, patterns)


# ============================================================
# Troubleshooting
# ============================================================

def detect_troubleshooting(text):
    """
    Detect concrete troubleshooting instructions.

    Generic phrases such as:
        'Here's what you can do'
        'We'd like to help'

    are deliberately NOT enough.

    We look for actual procedural instructions.
    """

    patterns = [
        # Explicit troubleshooting guidance
        r"\btroubleshooting steps\b",

        # Concrete actions
        r"\bplease check\b",
        r"\bplease try\b",
        r"\bplease use\b",
        r"\bplease follow\b",
        r"\btry to restart\b",
        r"\btry restarting\b",
        r"\btry resetting\b",
        r"\btry restoring\b",

        # Device operations
        r"\blong press\b",
        r"\bforce restart\b",
        r"\breset\b.*\bsettings\b",
        r"\brestore\b.*\bdevice\b",
        r"\brestore\b.*\bfactory\b",
        r"\buse iTunes\b",

        # Navigation / procedural steps
        r"\bgo to Settings\b",
        r"\bfollow these steps\b",
        r"\bcheck out these steps\b",

        # Concrete connection action
        r"\bmove closer to\b.*\brouter\b",

        # Troubleshooting phrased as a question.
        # 'restarting' is intentionally excluded because
        # 'have you tried restarting?' is annotated as
        # ASK_DIAGNOSTIC_QUESTION + RESTART_DEVICE.
        r"\bhave you tried\b.*\b(?:moving closer to|turning|resetting|restoring|using)\b",

        # Settings action
        r"\btoggle\b.*\b(?:on|off)\b",
        r"\benable\b.*\b(?:feature|option)\b",
        r"\bdisable\b.*\b(?:feature|option)\b",

        # Specific procedural solution
        r"\bcombine your songs\b.*\busing iTunes\b",

        # Article explicitly providing troubleshooting guidance
        r"\barticle\b.*\bfor troubleshooting\b",
    ]

    return contains_any(text, patterns)


# ============================================================
# Information / resources
# ============================================================

def detect_article(text):
    """
    Detect a non-DM support/resource link.

    A DM URL alone is NOT an article.
    A non-DM URL with clear resource, reference, workaround,
    specialist, or informational context is treated as a resource.
    """

    # Feedback and language-routing URLs are not treated as
    # support articles.
    if detect_feedback(text):
        return False

    if detect_language_support(text):
        return False

    if not has_resource_link(text):
        return False

    patterns = [
        # Explicit article / guide language
        r"\bcheck out this article\b",
        r"\bthis article\b",
        r"\bwe have an article\b",
        r"\barticle that can help\b",
        r"\bguide\b",
        r"\bthis guide\b",
        r"\buse this guide\b",
        r"\bcheck out this guide\b",

        # Learning / reference language
        r"\blearn how to\b",
        r"\blearn more here\b",
        r"\bhow to\b",
        r"\binstructions\b",
        r"\bsteps\b.*\bhere\b",
        r"\bcheck out these steps\b",
        r"\bhave a look\b.*\bhere\b",
        r"\bcheck this out\b",

        # Resource explicitly associated with workaround
        r"\bworkaround\b.*\bhere\b",
        r"\bworkaround for this here\b",
        r"\bwork around\b.*\bsoftware update\b",
        r"\bback up\b.*:",

        # Informational/reference links
        r"\bhere:\s*(?:https?://|www\.)",
        r"\bhere\s+(?:https?://|www\.)",
        r"\bcontact options here\b",
        r"\bcurrently selling, here\b",
        r"\bmodels.*currently selling.*here\b",

        # Specialist/resource routing
        r"\bexperts here\b",
        r"\bcontact.*here\b",
        r"\boptions here\b",
    ]

    return contains_any(text, patterns)


def detect_information(text):
    """Detect replies that mainly provide factual information."""

    patterns = [
        r"\bis operating as it should\b",
        r"\bno need to worry\b",
        r"\bis a microphone\b",
        r"\bis a speaker\b",
        r"\bit(?:'s| is) probably fraudulent\b",
        r"\bcan't be adjusted directly\b",
        r"\bcan't be adjusted\b",

        # Factual/educational guidance
        r"\brespond properly to and report phishing attempts\b",
    ]

    return contains_any(text, patterns)


# ============================================================
# Conversation actions
# ============================================================

def detect_diagnostic_question(text):
    """
    Detect questions intended to narrow down the cause,
    behavior, conditions, or state of a problem.

    A question asking only for static user/device details
    should normally be handled by ASK_FOR_DETAILS instead.
    """

    if "?" not in text:
        return False

    patterns = [
        # What happens / observed behavior
        r"\bwhat happens\b",
        r"\bwhat exactly happens\b",
        r"\bwhat happens when\b",
        r"\bwhat happens if\b",

        # Clarifying the actual problem/behavior
        r"\bwhat seems to be the issue\b",
        r"\bwhat(?:'s| is) going on\b",
        r"\bwhat issue\b",
        r"\bclarify the issue\b",
        r"\bwhat do you mean\b",

        # Testing a condition or state
        r"\bis the issue\b",
        r"\bis this happening\b",
        r"\bis it happening\b",
        r"\bis .* still\b",
        r"\bdoes this happen\b",
        r"\bdoes it happen\b",
        r"\bdoes the issue\b",
        r"\bdoes the problem\b",
        r"\bdoes it still\b",
        r"\bdo .* show up\b",
        r"\bdo .* appear\b",

        # Comparing behavior across conditions/devices
        r"\bon any other devices\b",
        r"\bon another device\b",
        r"\bwith any specific apps\b",
        r"\bconnected to\b.*\bor\b.*\bcreating\b",

        # Questions about what happens after/before an action
        r"\bafter\b.*\?",
        r"\bbefore\b.*\?",

        # Diagnostic yes/no questions
        r"\bhave you restarted\b",
        r"\bhave you checked\b",
        r"\bdid you try\b",
        r"\bdid you restart\b",
        r"\bdid you check\b",

        # Ability/behavior tests
        # 'send' is intentionally excluded so that
        # 'Can you send us a DM with your model...'
        # remains ASK_FOR_DETAILS + CONTACT_SUPPORT_DM.
        r"\bare you able to\b",
        r"\bcan you\b.*\b(?:connect|access|download|receive|open|use)\b",

        # Specific diagnostic questions
        r"\bwhich .* issue\b",
        r"\bwhere\b.*\bissue\b",

        # Narrowing down between two possible causes or scenarios
        r"\bonly begun\b.*\bsince\b",
        r"\bonly when\b",
        r"\bor only\b",
        r"\bor the\b.*\?",
        r"\bbetween\b.*\band\b",

        # Testing behavior after a previous action
        r"\bwhen you\b.*\?",
        r"\bwhen .* did you\b",
        r"\bafter .* did you\b",
        r"\bwhen .* see\b.*\?",

        # Asking what the user observes
        r"\bdo you see\b",
        r"\bdo you notice\b",
        r"\bdoes anything happen\b",
        r"\bwhat are you downloading\b",
        r"\bdo you have it connected to\b",

        # Specific example of describing the observed problem
        r"\bgive us an example of the issue\b",

        # When the issue started
        r"\bwhen did this start\b",

        # Country/region can be diagnostic context in support routing
        r"\bwhat country or region\b",

        # iOS version asked in a troubleshooting context
        r"\bwhat version of ios\b.*\bsettings\b",

        # Distinguishing connection/setup behavior
        r"\bconnecting to\b.*\bcreating\b",
        r"\bconnecting to\b.*\bor\b.*\bcreating\b",
    ]

    return contains_any(text, patterns)


def detect_details(text):
    """
    Detect requests for contextual information such as
    device, model, OS version, location, or additional details.
    """

    patterns = [
        # Device / model
        r"\bwhat model\b",
        r"\bwhich model\b",
        r"\bwhich device\b",
        r"\bwhat device\b",
        r"\btype of device\b",

        # Software version
        r"\bios version\b",
        r"\bversion number\b",
        r"\bexact version\b",
        r"\bwhat version\b",
        r"\bwhich version\b",

        # Location
        r"\bwhat country\b",
        r"\bwhich country\b",
        r"\bcountry.*located\b",
        r"\bregion.*located\b",
        r"\bwhat region\b",
        r"\bwhere you are\b",

        # Additional context
        r"\bmore details\b",
        r"\bspecific details\b",
        r"\bany steps tried\b",
        r"\bsteps.*tried\b",
        r"\bfill us in\b",
        r"\bgive us some details\b",
        r"\bwhat you're experiencing\b",
        r"\bwhat you(?:'re| are) noticing\b",

        # Explicit request to describe the issue
        r"\btell us.*\bissue\b",
        r"\btell us.*\bdetails\b",
        r"\bprovide us.*\bdetails\b",
        r"\bwhere you are seeing this\b",
        r"\bwhat.*started\b",

        # DM-specific requests for issue/context
        r"\bdm.*specific issue\b",
        r"\bdm.*more information\b",
        r"\bdm.*details\b",
        r"\bdm.*what(?:'s| is) going on\b",
        r"\bdm.*what you(?:'re| are) experiencing\b",
        r"\bdm.*type of device\b",
        r"\bdm.*model\b",
        r"\bdm.*ios version\b",
        r"\bdm.*version\b",
    ]

    return contains_any(text, patterns)


def detect_dm(text):
    """Detect an invitation/request to continue support via Direct Message."""

    invitation_patterns = [
        # Explicit DM requests
        r"\bdm us\b",
        r"\bplease dm us\b",
        r"\bsend us a dm\b",
        r"\bsend us.*\bdm\b",
        r"\bshoot us a dm\b",
        r"\bshooting us a dm\b",
        r"\bshoot us.*\bdm\b",
        r"\bdirect message us\b",

        # Tell / contact / continue through DM
        r"\btell us in a dm\b",
        r"\blet us know in dm\b",
        r"\blet us know in direct message\b",
        r"\bjoin us in dm\b",
        r"\bmeet us in dm\b",
        r"\bmeet up in dm\b",
        r"\bmeet us there\b",
        r"\btake this to dm\b",
        r"\breach out to us via dm\b",
        r"\bwork with you via dm\b",
        r"\bwork with you via twitter'?s dm system\b",
        r"\bpartner together in dm\b",

        # Invitation language
        r"\binvitation to join us there\b",
        r"\binvite.*to dm\b",
        r"\bin dm so we can\b",
        r"\bin dm to\b",

        # DM + request for information
        r"\bdm us.*details\b",
        r"\bdm us.*information\b",
        r"\bdm us.*response\b",
        r"\bdm us.*experiencing\b",
        r"\bdm.*specific issue\b",
        r"\bdm.*more information\b",
        r"\bdm.*exact version\b",
        r"\bdm.*what you are experiencing\b",
        r"\bdm.*type of device\b",
        r"\bplease dm us.*what\b",

        # "DM + bare resource link" — common AppleSupport pattern
        r"\bdm\b",

        # Existing AppleSupport phrasing
        r"\bfollow us to dm\b",
        r"\bvia dm\b",
    ]

    acknowledgement_patterns = [
        r"\breceived your dm\b",
        r"\bgot your dm\b",
        r"\bwe'?ve got your dm\b",
        r"\bthanks for the dm\b",
        r"\bthank you for the dm\b",
        r"\bcontinue with you there\b",
        r"\bcontinue with you in dm\b",
    ]

    # Acknowledgement of an already-started DM is not a new DM invitation.
    if contains_any(text, acknowledgement_patterns):
        return False

    # A known AppleSupport DM URL is strong evidence of a DM action,
    # even when the surrounding text does not explicitly say "DM".
    if any(is_dm_url(url) for url in extract_urls(text)):
        return True

    return contains_any(text, invitation_patterns)


# ============================================================
# Routing
# ============================================================

def detect_specialist(text):
    """Detect routing to a specialist/team."""

    patterns = [
        r"\bphotos experts?\b",
        r"\bsales team\b",
        r"\bspecialist\b",
        r"\bexperts?\b.*\bassist\b",
        r"\bteam\b.*\bbest\b.*\bassist\b",
    ]

    return contains_any(text, patterns)


def detect_feedback(text):
    """Detect feedback submission/routing."""

    patterns = [
        r"\bwelcome feedback\b",
        r"\bsubmit.*feedback\b",
        r"\bsend.*feedback\b",
        r"\bshare.*feedback\b",
        r"\bfeedback\b.*\bhere\b",
        r"\bsubmit yours\b",
    ]

    return contains_any(text, patterns)


def detect_language_support(text):
    """Detect language-specific support routing."""

    patterns = [
        r"\bsupport via twitter in english\b",
        r"\bget help in spanish\b",
        r"\bhelp in spanish\b",
        r"\bsupport in\b.*\blanguage\b",
        r"\blanguage support\b",
    ]

    return contains_any(text, patterns)


# ============================================================
# Main extraction function
# ============================================================

def extract_resolution_actions(reply):
    """
    Extract all meaningful actions from an AppleSupport reply.

    Multiple actions may be returned for one reply.
    """

    text = normalize_text(reply)

    if not text:
        return ["NO_CLEAR_ACTION"]

    actions = []

    # Resolution
    if detect_announce_fix_update(text):
        actions.append("ANNOUNCE_FIX_UPDATE")

    if detect_update_software(text):
        actions.append("UPDATE_SOFTWARE")

    if detect_restart(text):
        actions.append("RESTART_DEVICE")

    if detect_settings(text):
        actions.append("CHECK_SETTINGS")

    if detect_connection(text):
        actions.append("CHECK_CONNECTION")

    if detect_account(text):
        actions.append("CHECK_ACCOUNT")

    if detect_app_store(text):
        actions.append("CHECK_APP_STORE")

    if detect_battery(text):
        actions.append("CHECK_BATTERY")

    if detect_workaround(text):
        actions.append("PROVIDE_WORKAROUND")

    # Information / resources
    if detect_troubleshooting(text):
        actions.append("PROVIDE_TROUBLESHOOTING")

    if detect_information(text):
        actions.append("PROVIDE_INFORMATION")

    if detect_article(text):
        actions.append("PROVIDE_ARTICLE")

    # Conversation
    if detect_diagnostic_question(text):
        actions.append("ASK_DIAGNOSTIC_QUESTION")

    if detect_details(text):
        actions.append("ASK_FOR_DETAILS")

    if detect_dm(text):
        actions.append("CONTACT_SUPPORT_DM")

    # Routing
    if detect_specialist(text):
        actions.append("ROUTE_TO_SPECIALIST")

    if detect_feedback(text):
        actions.append("ROUTE_FEEDBACK")

    if detect_language_support(text):
        actions.append("ROUTE_LANGUAGE_SUPPORT")

    # Remove duplicates
    actions = list(dict.fromkeys(actions))

    if not actions:
        return ["NO_CLEAR_ACTION"]

    return actions


# ============================================================
# Batch processing
# ============================================================

def main():

    print("=" * 70)
    print("SupportLens AI - Resolution Action Extraction")
    print("=" * 70)

    print(f"\nInput:  {INPUT_FILE}")
    print(f"Output: {OUTPUT_FILE}")

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Input file not found: {INPUT_FILE}"
        )

    df = pd.read_csv(INPUT_FILE)

    print(f"\nLoaded {len(df)} historical replies.")

    reply_column = "support_reply"

    if reply_column not in df.columns:
        raise ValueError(
            f"Expected '{reply_column}' column. "
            f"Available columns: {list(df.columns)}"
        )

    df["extracted_actions"] = df[reply_column].apply(
        lambda x: " | ".join(
            extract_resolution_actions(x)
        )
    )

    df["action_count"] = df["extracted_actions"].apply(
        lambda x: len(x.split(" | "))
    )

    df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    # --------------------------------------------------------
    # Frequency
    # --------------------------------------------------------

    action_counts = {}

    for actions in df["extracted_actions"]:

        for action in actions.split(" | "):

            action_counts[action] = (
                action_counts.get(action, 0) + 1
            )

    print("\n" + "=" * 70)
    print("ACTION FREQUENCY")
    print("=" * 70)

    for action, count in sorted(
        action_counts.items(),
        key=lambda x: x[1],
        reverse=True
    ):
        print(f"{action:28} {count:>4}")

    print("\n" + "=" * 70)
    print("Extraction complete.")
    print(f"Saved: {OUTPUT_FILE}")
    print("=" * 70)


if __name__ == "__main__":
    main()