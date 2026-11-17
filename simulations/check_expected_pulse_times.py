import ROOT

# Open ROOT file
filename = "/scratch/midway3/darcynewmark/simulation/testing/run19000/IceFinal_19000_allTree.root"

root_file = ROOT.TFile.Open(filename)
tree = root_file.Get("passTree")

if not tree:
    raise RuntimeError("Could not find passTree!")

# Loop over events
for i_event in range(tree.GetEntries()):
    tree.GetEntry(i_event)

    print(f"\n{'='*60}")
    print(f"Event {i_event}")
    print(f"{'='*60}")

    # Loop over detectors
    for i_detector, detector in enumerate(tree.detectorEvents):

        print(f"\nDetector {i_detector}")

        # Loop over showers
        for i_shower, pulse_times in enumerate(detector.expectedPulseTimes):

            print(f"  Shower {i_shower}: {list(pulse_times)}")

            # Print individual pulse times
            for i_pulse, time in enumerate(pulse_times):
                print(f"    Pulse {i_pulse}: {time:.6f}")

root_file.Close()

