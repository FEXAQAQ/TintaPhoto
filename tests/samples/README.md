# Test photos

The photos themselves are not in the repository (`.gitignore` keeps every RAW
and JPEG out of it); this file is, so that an empty folder still says what
belongs in it. Without them a part of the suite skips, and
`tests/test_reference.py` - the comparison PLAN.md section 11 asks for -
checks nothing at all.

## What the suite looks for

| Expected here | Used by | Missing means |
| --- | --- | --- |
| `_DSC2828.ARW` - uncompressed RAW | `tests/test_reference.py` | 4 reference cases skip |
| `_DSC2823.ARW` - compressed RAW | `tests/test_reference.py` | 4 reference cases skip |
| `_DSC2828.JPG` - the Adobe RGB JPEG of the same shot | `tests/test_reference.py` | 4 reference cases skip |
| any `*.ARW` | `test_decoder`, `test_cache`, `test_exporter`, `test_metadata`, `tests/benchmark.py` | those tests skip |
| any `*.JPG` | `test_decoder` | the JPEG tests skip |
| any `*.RAF` | `test_metadata` | the two tests for a RAW that is not a TIFF underneath skip |
| a RAW the camera recorded rotated | `test_decoder` | the three orientation tests skip |
| a RAW with a blown sky, over 0.1% of the frame clipped | `test_decoder` | the clipping mask is only exercised on specular highlights |

The three photos named above are the reference set, and the names matter: each
one has four stored fingerprints in `tests/reference/`, one per case. Changing
which photos those are means recording the fingerprints again, which resets
the baseline the comparison holds the pipeline to - see the note at the top of
`tests/test_reference.py`.

The rest of the folder is open: any ARW and any JPEG will do, and more of them
simply means more coverage. A photo the camera recorded rotated and a photo
with a blown sky are each worth having, because a few tests have nothing to
run on without them.

## Recording the fingerprints

After putting the reference photos in place, or when a change to
`core/pipeline.py` is meant to move the colours:

```powershell
$env:TINTAPHOTO_WRITE_REFERENCE = "1"; pytest tests/test_reference.py; Remove-Item env:TINTAPHOTO_WRITE_REFERENCE
```

Then commit `tests/reference/` and say in the commit message why the colours
moved.
