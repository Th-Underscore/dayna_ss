#!/usr/bin/env python3
"""Analyze per-turn LLM call counts from the soak log."""
import re
import sys
from collections import defaultdict

def main():
    log_path = sys.argv[1] if len(sys.argv) > 1 else "runs/logs/scifi_heist_gemma4_s520.log"
    
    turns = {}
    current_turn = -1
    
    turn_pat = re.compile(r'--- turn (\d+)/\d+ ---')
    call_pat = re.compile(r'local-call (\S+)/generate.*?in=(\d+) out=(\d+)')
    node_pat = re.compile(r"Processing Schema Node.*?at '([^']+)'")
    # Scene detection: "new scene" in the DSS output
    scene_pat = re.compile(r'(?:on_new_scene|new scene detected|Scene boundary)')
    
    with open(log_path, 'r', errors='replace') as f:
        for line in f:
            m = turn_pat.search(line)
            if m:
                current_turn = int(m.group(1))
                if current_turn not in turns:
                    turns[current_turn] = {'calls': [], 'nodes': [], 'scene': False}
                continue
            
            if current_turn < 0:
                continue
            
            m = call_pat.search(line)
            if m:
                turns[current_turn]['calls'].append({
                    'phase': m.group(1),
                    'in': int(m.group(2)),
                    'out': int(m.group(3))
                })
                continue
            
            m = node_pat.search(line)
            if m:
                turns[current_turn]['nodes'].append(m.group(1))
                continue
            
            if scene_pat.search(line):
                turns[current_turn]['scene'] = True
    
    print(f"Total turns: {len(turns)}")
    scene_turns = [t for t in turns if turns[t]['scene']]
    non_scene = [t for t in turns if not turns[t]['scene']]
    print(f"Scene turns: {len(scene_turns)}  Non-scene: {len(non_scene)}")
    
    # Summary table
    print(f"\n{'Turn':>5} {'Scn':>4} {'Calls':>6} {'In-tok':>8} {'Nodes':>6}")
    print("-" * 38)
    for tn in sorted(turns.keys()):
        t = turns[tn]
        s = "Y" if t['scene'] else ""
        print(f"{tn:>5} {s:>4} {len(t['calls']):>6} {sum(c['in'] for c in t['calls']):>8} {len(t['nodes']):>6}")
    
    # Average calls
    non_scene_calls = [len(turns[t]['calls']) for t in non_scene]
    scene_calls = [len(turns[t]['calls']) for t in scene_turns]
    if non_scene_calls:
        print(f"\nNon-scene avg calls: {sum(non_scene_calls)/len(non_scene_calls):.1f} (min {min(non_scene_calls)}, max {max(non_scene_calls)})")
    if scene_calls:
        print(f"Scene avg calls: {sum(scene_calls)/len(scene_calls):.1f} (min {min(scene_calls)}, max {max(scene_calls)})")
    
    # Show node breakdown for a specific turn
    show_turn = int(sys.argv[2]) if len(sys.argv) > 2 else None
    if show_turn is None:
        # Default: show first non-scene turn with moderate call count
        candidates = sorted(non_scene, key=lambda t: abs(turns[t]['calls'] - 20))
        show_turn = candidates[0] if candidates else (sorted(turns.keys())[1] if len(turns) > 1 else 0)
    
    print(f"\n\n=== FULL NODE LIST: TURN {show_turn} ({len(turns[show_turn]['nodes'])} nodes, {len(turns[show_turn]['calls'])} calls) ===")
    t = turns[show_turn]
    for i, node in enumerate(t['nodes']):
        print(f"  {i+1:3d} {node}")

if __name__ == '__main__':
    main()